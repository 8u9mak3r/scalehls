//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;


/// Op fuse strategies:
/// 1. Transposes are to be fused by their matmul users
/// 2. Reshape-type ops are to be fused by their reduction-type users(matmul, conv, reduce_sum, reduce_max, etc.)
/// 3. Elementwise-type and broadcast-type Ops are to be fused by their users (broadcast-type ops can always be treated as elementwise ones)
/// Finally, be greedy and aggressive!


namespace {
/// This pattern will outline ops with the specified type.
template <typename InterfaceType>
struct OutlineRootInterface : public OpInterfaceRewritePattern<InterfaceType> {
  using OpInterfaceRewritePattern<InterfaceType>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(InterfaceType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();
    fuseOpsIntoTask({op}, rewriter);
    return success();
  }
};
} // namespace

namespace {
/// This pattern will outline ops with the specified type.
template <typename OpType>
struct OutlineRootOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();

    fuseOpsIntoTask({op}, rewriter);
    return success();
  }
};
} // namespace

namespace {
/// This pattern will outline ops with the specified type.
struct OutlineRootReductionTypeGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();

    if (!isReductionTypeGenericOp(op)) return failure();

    fuseOpsIntoTask({op}, rewriter);
    return success();
  }
};
} // namespace

namespace {
/// This pattern will outline ops with the specified type.
// template <typename OpType>
struct OutlineRootFinalOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();

    /// By default, we assume the terminating operation of the whole workload will be generics
    /// besides matmuls, convs and reductions
    /// TODO: What about reshape-type operations
    if (op->hasOneUse()) {
      for (auto user : op->getUsers()) {
        if (isa<DispatchOp>(user->getParentOp())) {
          fuseOpsIntoTask({op}, rewriter);
          return success();
        }
      }
    }

    return failure();
  }
};
} // namespace

namespace {
/// This pattern will forward fuse ops with the specified type.
template <typename OpType>
struct ForwardFuseOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();
    auto DT = DominanceInfo();

    auto builder = OpBuilder(rewriter.getContext());
    bool noTaskUsers = true;
                      
    /// if the result of the operation to be forward fused has multiple consumers
    /// then generate one copy for each and replace the corresponding use of the result
    for (auto& use : llvm::make_early_inc_range(op->getUses())) {
      if (auto task = dyn_cast<TaskOp>(use.getOwner()->getParentOp())) {
        noTaskUsers = false;
        builder.setInsertionPoint(op);
        auto clone = cast<OpType>(builder.clone(*op));
        
        auto cloneResult = clone->getResult(0);

        use.set(cloneResult);

        fuseOpsIntoTask({clone, task}, rewriter, /*insertToLastOp=*/true);
      }
    }

    if (noTaskUsers) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
/// This pattern will backward fuse ops with the specified type.
template <typename OpType>
struct BackwardFuseOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<TaskOp>())
      return failure();
    auto DT = DominanceInfo();

    // Find all task defining ops.
    SmallVector<TaskOp, 4> taskDefOps;
    for (auto operand : op->getOperands())
      if (auto task = operand.template getDefiningOp<TaskOp>())
        taskDefOps.push_back(task);
    if (taskDefOps.empty())
      return failure();

    // We always select the dominated task as the target to fuse.
    // FIXME: Check there's no intervening ops in between.
    llvm::sort(taskDefOps, [&](auto a, auto b) { return DT.dominates(a, b); });
    fuseOpsIntoTask({taskDefOps.back(), op}, rewriter, /*insertToLastOp=*/true);
    return success();
  }
};
} // namespace

namespace {
/// Forward fuse generic tensor copies.
struct ForwardFuseGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    bool matched = false;
    auto body = op.getBody();

    if (op.getNumOutputs() == 1 &&
        llvm::hasSingleElement(body->getOperations())) {
      auto output = body->getTerminator()->getOperand(0);

      // Copy from input to output.
      if (op.getNumInputs() == 1 && output == body->getArgument(0))
        matched = true;

      // Copy from constant to output.
      if (output.getDefiningOp<tosa::ConstOp>() ||
          output.getDefiningOp<arith::ConstantOp>())
        matched = true;
    }

    if (matched) {
      auto pattern = ForwardFuseOp<linalg::GenericOp>(getContext());
      return pattern.matchAndRewrite(op, rewriter);
    }
    return failure();
  }
};
} // namespace

namespace {
/// Backward fuse generic tensor elementwise ops.
struct BackwardFuseGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (isElementwiseGenericOp(op)) {
      auto pattern = BackwardFuseOp<linalg::GenericOp>(getContext());
      return pattern.matchAndRewrite(op, rewriter);
    }
    return failure();
  }
};
} // namespace

static void
populateForwardBackwardFusePatterns(mlir::RewritePatternSet &patterns) {
  auto context = patterns.getContext();

  /// Normally, generics are all elementwise-type ones and broadcast-type ones excluding reduction-type ones, which are all fusable
  patterns.add<ForwardFuseOp<linalg::GenericOp>>(context);
  patterns.add<ForwardFuseOp<linalg::FillOp>>(context);
  patterns.add<ForwardFuseOp<tensor::EmptyOp>>(context);
  patterns.add<ForwardFuseOp<tensor::PadOp>>(context);
  patterns.add<ForwardFuseOp<tensor::ExpandShapeOp>>(context);
  patterns.add<ForwardFuseOp<tensor::CollapseShapeOp>>(context);
  patterns.add<ForwardFuseOp<tensor::InsertSliceOp>>(context);
  patterns.add<ForwardFuseOp<tensor::ExtractSliceOp>>(context);
}

namespace {
struct CreateDataflowFromLinalg
    : public CreateDataflowFromLinalgBase<CreateDataflowFromLinalg> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    auto builder = OpBuilder(context);

    dispatchBlock(&func.front());

    // Collect all empty tensors in the function and localize them to uses.
    SmallVector<Operation *, 16> empties;
    func.walk([&](tensor::EmptyOp op) { empties.push_back(op); });
    for (auto empty : empties) {
      for (auto &use : llvm::make_early_inc_range(empty->getUses())) {
        builder.setInsertionPoint(use.getOwner());
        auto cloneEmpty = cast<tensor::EmptyOp>(builder.clone(*empty));
        use.set(cloneEmpty.getResult());
      }
      empty->erase();
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<OutlineRootInterface<linalg::ConvolutionOpInterface>>(context);
    patterns.add<OutlineRootInterface<linalg::ContractionOpInterface>>(context);
    patterns.add<OutlineRootReductionTypeGenericOp>(context);
    patterns.add<OutlineRootFinalOp>(context);
    populateForwardBackwardFusePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    patterns.clear();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createCreateDataflowFromLinalgPass() {
  return std::make_unique<CreateDataflowFromLinalg>();
}

