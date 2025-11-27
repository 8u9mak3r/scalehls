#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include <cassert>

using namespace mlir;
using namespace scalehls;
using namespace hls;

/// Assume we got a couple of depending tasks here:
/// hls.dataflow.task %0 : (s0xs1xs2xf32) = {...} --> %1 : (xf32) {...ops on %0...}
/// We call the matmul/conv/reduction/etc op in task %1 the actual consumer, which is the fundament of the task(Every task has an actual consumer)
/// Rationale here is: 
/// Task %1 may perform a reshape op on the result of task %0 beforehand, causing the shape of the tensor fed to the actual consumer
/// in %1 differing with that of %0's result.
/// In this case, we uplift the very reshape op from %1 to %0 and place it just right before the terminating yield op of %0
/// This strategy will help the future's grand fusion into one linalg.generic of all trivial ops within each task

namespace {
/// This pattern will outline ops with the specified type.
struct WalkOverTasksAndUpliftReshapeOps : public OpRewritePattern<hls::TaskOp> {
  using OpRewritePattern<hls::TaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hls::TaskOp op,
                                PatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::Operation*, 64> opsQueue;

    int defineUsePaths[64] = {-1, 0};
    auto iter = opsQueue.begin();
    int idx = 0;

    for (auto result : op.getResults()) {
      if (result.use_empty()) continue;

      auto producedShape = dyn_cast<mlir::RankedTensorType>(result.getType()).getShape();
      result.getDefiningOp()->dump();
      opsQueue.push_back(result.getDefiningOp());

      while (iter != opsQueue.end()) {
        for (auto& use : llvm::make_early_inc_range((*iter)->getUses())) {
          if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(*iter)) {
            auto targetShape = dyn_cast<mlir::RankedTensorType>((*iter)->getResult(0).getType()).getShape();
            for (int64_t dim : producedShape) llvm::dbgs() << dim << " ";
            llvm::dbgs() << "\n";
            for (int64_t dim : targetShape) llvm::dbgs() << dim << " ";
            llvm::dbgs() << "\n";
            if (!isSimilarShapes(producedShape, targetShape)) {
              llvm::dbgs() << "Oops\n";
              break;
            }
          }
          auto p = use.getOwner();
          p->dump();
          if (isa<hls::YieldOp>(p)) continue;
          opsQueue.push_back(p);
          
          defineUsePaths[++idx] = iter - opsQueue.begin();
        }
        // llvm::dbgs() << "\n";
        ++iter;
      }

      // SmallVector<mlir::Operation*, 8> path;
      // for (int i = idx; i >= 0; i -= 1) {
      //   // llvm::dbgs() << "\n\n\n";
      //   iter = opsQueue.begin() + i;
      //   // if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(*iter)) continue;
      //   // auto targetShape = dyn_cast<mlir::RankedTensorType>((*iter)->getResult(0).getType()).getShape();
      //   // if (isSimilarShapes(producedShape, targetShape)) continue;
      //   (*iter)->dump();
      //   // while (iter != opsQueue.begin()) {
      //   //   iter = opsQueue.begin() + defineUsePaths[iter - opsQueue.begin()];
      //   //   path.push_back(*iter);
      //   //   (*iter)->dump();
      //   // }
      //   // path.clear();
      // }

      llvm::dbgs() << "\n\n\n";
    }

    return success();
  }
};
} // namespace


namespace {
struct ReshapeUpliftBetweenTasks
    : public ReshapeUpliftBetweenTasksBase<ReshapeUpliftBetweenTasks> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<WalkOverTasksAndUpliftReshapeOps>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), {false, true, 1L});
    patterns.clear();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createReshapeUpliftBetweenTasksPass() {
  return std::make_unique<ReshapeUpliftBetweenTasks>();
}
