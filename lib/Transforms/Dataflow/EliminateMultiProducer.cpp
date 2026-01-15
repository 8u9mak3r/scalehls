//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {
struct BufferizeMultiProducer : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    DominanceInfo DT(schedule);
    auto loc = rewriter.getUnknownLoc();
    bool hasChanged = false;

    SmallVector<Value> buffers;
    for (auto bufferOp : schedule.getOps<BufferOp>())
      buffers.push_back(bufferOp);

    for (auto buffer : buffers) {
      SmallVector<NodeOp, 4> producers(getProducers(buffer));
      if (producers.size() <= 1)
        continue;
      hasChanged = true;

      // Drop the dominating/leading producer, which doesn't need to be
      // transformed.
      llvm::sort(producers, [&](NodeOp a, NodeOp b) {
        return DT.properlyDominates(b, a);
      });
      producers.pop_back();

      for (auto node : producers) {
        auto bufferIdx = llvm::find(node.getOutputs(), buffer) -
                         node.getOutputs().begin() + node.getNumInputs();
        assert(bufferIdx < node.getNumInputs() + node.getNumOutputs());
        rewriter.setInsertionPoint(node);
        auto newBuffer = rewriter.create<hls::BufferOp>(loc, buffer.getType());
        node.setOperand(bufferIdx, newBuffer);

        buffer.replaceUsesWithIf(newBuffer, [&](OpOperand &use) {
          if (auto user = dyn_cast<NodeOp>(use.getOwner()))
            return DT.properlyDominates(node, user);
          return false;
        });

        auto newBufferArg = node.getBody().getArgument(bufferIdx);
        auto readUses = llvm::make_filter_range(
          newBufferArg.getUses(),
          [](OpOperand& use) { return isRead(use); }
        );
        auto writeUses = llvm::make_filter_range(
          newBufferArg.getUses(),
          [](OpOperand& use) { return isWritten(use); }
        );

        // We assume that these extra producer nodes are either feature a read(affine.load) and a write(affine.store) 
        // on the output buffer, reduction nodes such as matmul/conv/sum in typical, or merely feature a single write
        if (llvm::hasSingleElement(readUses) && llvm::hasSingleElement(writeUses)) {
          auto read = dyn_cast<AffineLoadOp>(readUses.begin()->getOwner());
          auto write = dyn_cast<AffineStoreOp>(writeUses.begin()->getOwner());
          // In the former case, we assume that the load op and the store op are performed on the exact same position of the buffer
          // which is, actually, almost always this case.
          // Also the read should dominate the write, needless to say
          assert(DT.properlyDominates(read.getOperation(), write.getOperation())
            && "A read on an output buffer should dominate the matching write within one node."
          );
          assert(read.getMapOperands() == write.getMapOperands()
            && "The code pattern in which read/write are performed on different positions is not supported."
          );

          // Use the old buffer as an input to be read, while the new one as an output to be written into
          auto bufferArg = node.getBody().insertArgument(
            bufferIdx, buffer.getType(), buffer.getLoc()
          );
          // readUses.begin()->set(bufferArg);

          SmallVector<Value> nodeInputs = node.getInputs();
          nodeInputs.push_back(buffer);
          SmallVector<unsigned int> inputTaps = node.getInputTapsAsInt();
          inputTaps.push_back(0);

          auto newNode = rewriter.create<hls::NodeOp>(
            node.getLoc(),
            nodeInputs,
            node.getOutputs(),
            node.getParams(),
            inputTaps,
            node.getLevelAttr()
          );
          rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(), newNode.getBody().begin());
          rewriter.eraseOp(node);

          AffineLoopBand band;
          getLoopIVs(*read.getOperation(), &band);

          llvm::SmallDenseSet<Value> depInductionVars;
          for (auto loop : band)
            depInductionVars.insert(loop.getInductionVar());

          auto noEscapeIndex = true;
          for (auto operand : read.getMapOperands())
            if (!depInductionVars.erase(operand))
              noEscapeIndex = false;

          if (noEscapeIndex) {
            SmallVector<AffineExpr, 4> ifExprs;
            SmallVector<bool, 4> ifEqFlags;
            SmallVector<Value, 4> ifOperands;
            unsigned dim = 0;
            for (auto iv : depInductionVars) {
              auto loop = getForInductionVarOwner(iv);

              // Create all components required by constructing if operation.
              if (loop.hasConstantLowerBound()) {
                ifExprs.push_back(rewriter.getAffineDimExpr(dim++) -
                                  loop.getConstantLowerBound());
                ifOperands.push_back(loop.getInductionVar());
              } else {
                // Non-constant case requires to integrate the bound affine
                // expression and operands into the condition integer set.
                auto lowerExpr = loop.getLowerBoundMap().getResult(0);
                auto lowerOperands = loop.getLowerBoundOperands();
                SmallVector<AffineExpr, 4> newDims;
                for (unsigned i = 0, e = lowerOperands.size(); i < e; ++i)
                  newDims.push_back(rewriter.getAffineDimExpr(i + dim + 1));
                lowerExpr = lowerExpr.replaceDimsAndSymbols(newDims, {});

                ifExprs.push_back(rewriter.getAffineDimExpr(dim) - lowerExpr);
                ifOperands.push_back(loop.getInductionVar());
                ifOperands.append(lowerOperands.begin(), lowerOperands.end());
                dim += lowerOperands.size() + 1;
              }
              ifEqFlags.push_back(true);

              rewriter.setInsertionPoint(read);
              auto value = rewriter.create<mlir::AffineLoadOp>(
                  read.getLoc(), bufferArg, read.getAffineMap(), read.getMapOperands());

              if (!ifExprs.empty()) {
                auto ifCondition = IntegerSet::get(dim, 0, ifExprs, ifEqFlags);
                auto ifOp = rewriter.create<AffineIfOp>(
                    read.getLoc(), ifCondition, ifOperands, false);
                rewriter.setInsertionPointToStart(ifOp.getThenBlock());
              }

              rewriter.create<mlir::AffineStoreOp>(
                  read.getLoc(), value, newBufferArg, read.getAffineMap(), read.getMapOperands());
            }
          }
        } else {
          // In the latter case, nothing else needs to be done
          // Yet this pass doesn't support a third read/write pattern
          if (readUses.empty() && llvm::hasSingleElement(writeUses)) continue;
          llvm_unreachable("Abnormal read & write pattern occurred!\n");
        }
      }
    }

    

    // Verification
    for (auto bufferOp : schedule.getOps<BufferOp>())
      if (!llvm::hasSingleElement(getProducers(bufferOp)))
        return schedule.emitOpError("Failed to bufferize multiple producers!\n");

    return success(hasChanged);
  }
};
} // namespace

namespace {
struct MergeMultiProducer : public OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleOp schedule,
                                PatternRewriter &rewriter) const override {
    DominanceInfo DT(schedule);
    bool hasChanged = false;

    SmallVector<Value> externalBuffers;
    for (auto arg : schedule.getBody().getArguments())
      externalBuffers.push_back(arg);

    for (auto buffer : externalBuffers) {
      SmallVector<NodeOp> producers(getProducers(buffer));
      if (producers.size() <= 1)
        continue;

      llvm::sort(producers, [&](NodeOp a, NodeOp b) {
        return DT.properlyDominates(a, b);
      });

      auto allNodes = SmallVector<NodeOp>(schedule.getOps<NodeOp>().begin(),
                                          schedule.getOps<NodeOp>().end());
      auto ptr = llvm::find(allNodes, producers.front());
      if (llvm::any_of(llvm::enumerate(producers), [&](auto node) {
            return node.value() != *std::next(ptr, node.index());
          }))
        continue;

      fuseNodeOps(producers, rewriter);
      hasChanged = true;
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct EliminateMultiProducer
    : public EliminateMultiProducerBase<EliminateMultiProducer> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<BufferizeMultiProducer>(context);
    patterns.add<MergeMultiProducer>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createEliminateMultiProducerPass() {
  return std::make_unique<EliminateMultiProducer>();
}


// //===----------------------------------------------------------------------===//
// //
// // Copyright 2020-2021 The ScaleHLS Authors.
// //
// //===----------------------------------------------------------------------===//

// #include "mlir/IR/Dominance.h"
// #include "mlir/IR/IntegerSet.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "scalehls/Transforms/Passes.h"
// #include "scalehls/Transforms/Utils.h"

// using namespace mlir;
// using namespace scalehls;
// using namespace hls;

// namespace {
// struct BufferMultiProducer : public OpRewritePattern<ScheduleOp> {
//   using OpRewritePattern<ScheduleOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(ScheduleOp schedule,
//                                 PatternRewriter &rewriter) const override {
//     DominanceInfo DT(schedule);
//     auto loc = rewriter.getUnknownLoc();
//     bool hasChanged = false;

//     SmallVector<Value> buffers;
//     for (auto bufferOp : schedule.getOps<BufferOp>())
//       buffers.push_back(bufferOp);

//     for (auto buffer : buffers) {
//       SmallVector<NodeOp, 4> producers(getProducers(buffer));
//       if (producers.size() <= 1)
//         continue;
//       hasChanged = true;

//       // Drop the dominating/leading producer, which doesn't need to be
//       // transformed.
//       llvm::sort(producers, [&](NodeOp a, NodeOp b) {
//         return DT.properlyDominates(b, a);
//       });
//       producers.pop_back();

//       for (auto node : producers) {
//         auto newInputs = SmallVector<Value>(node.getInputs());
//         SmallVector<unsigned> newInputTaps(node.getInputTapsAsInt());
//         rewriter.setInsertionPoint(node);

//         // Create a new buffer and write to them instead of the original buffer.
//         // The original buffer will be passed into the new node as inputs.
//         newInputs.push_back(buffer);
//         newInputTaps.push_back(0);
//         auto newBuffer = rewriter.create<BufferOp>(loc, buffer.getType());
//         auto bufferIdx = llvm::find(node.getOutputs(), buffer) -
//                          node.getOutputs().begin() + node.getNumInputs();
//         node.setOperand(bufferIdx, newBuffer);

//         buffer.replaceUsesWithIf(newBuffer, [&](OpOperand &use) {
//           if (auto user = dyn_cast<NodeOp>(use.getOwner()))
//             return DT.properlyDominates(node, user);
//           return false;
//         });

//         // Create a new node and erase the original one.
//         auto newNode = rewriter.create<NodeOp>(
//             node.getLoc(), newInputs, node.getOutputs(), node.getParams(),
//             newInputTaps, node.getLevelAttr());
//         rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(),
//                                     newNode.getBody().end());
//         rewriter.eraseOp(node);

//         // Insert new arguments for the original buffer.
//         rewriter.setInsertionPointToStart(&newNode.getBody().front());
//         auto newBufferArg = newNode.getBody().getArgument(bufferIdx);
//         auto bufferArg = newNode.getBody().insertArgument(
//             newNode.getNumInputs() - 1, newBufferArg.getType(),
//             newBufferArg.getLoc());

//         // If the only read user of the buffer is affine load, we can avoid to
//         // create a redundant data copy.
//         auto readUses = llvm::make_filter_range(
//             newBufferArg.getUses(), [](OpOperand &use) { return isRead(use); });
//         if (llvm::hasSingleElement(readUses))
//           if (auto read = dyn_cast<mlir::AffineReadOpInterface>(
//                   readUses.begin()->getOwner())) {
//             // We need to make sure all the indices of the affine load are known
//             // loop induction variables and meanwhile the load has identity
//             // memory access map.
//             AffineLoopBand band;
//             getLoopIVs(*read, &band);

//             llvm::SmallDenseSet<Value> depInductionVars;
//             for (auto loop : band)
//               depInductionVars.insert(loop.getInductionVar());

//             auto noEscapeIndex = true;
//             for (auto operand : read.getMapOperands())
//               if (!depInductionVars.erase(operand))
//                 noEscapeIndex = false;

//             if (noEscapeIndex && read.getAffineMap().isIdentity()) {
//               SmallVector<AffineExpr, 4> ifExprs;
//               SmallVector<bool, 4> ifEqFlags;
//               SmallVector<Value, 4> ifOperands;
//               unsigned dim = 0;
//               for (auto iv : depInductionVars) {
//                 auto loop = getForInductionVarOwner(iv);

//                 // Create all components required by constructing if operation.
//                 if (loop.hasConstantLowerBound()) {
//                   ifExprs.push_back(rewriter.getAffineDimExpr(dim++) -
//                                     loop.getConstantLowerBound());
//                   ifOperands.push_back(loop.getInductionVar());
//                 } else {
//                   // Non-constant case requires to integrate the bound affine
//                   // expression and operands into the condition integer set.
//                   auto lowerExpr = loop.getLowerBoundMap().getResult(0);
//                   auto lowerOperands = loop.getLowerBoundOperands();
//                   SmallVector<AffineExpr, 4> newDims;
//                   for (unsigned i = 0, e = lowerOperands.size(); i < e; ++i)
//                     newDims.push_back(rewriter.getAffineDimExpr(i + dim + 1));
//                   lowerExpr = lowerExpr.replaceDimsAndSymbols(newDims, {});

//                   ifExprs.push_back(rewriter.getAffineDimExpr(dim) - lowerExpr);
//                   ifOperands.push_back(loop.getInductionVar());
//                   ifOperands.append(lowerOperands.begin(), lowerOperands.end());
//                   dim += lowerOperands.size() + 1;
//                 }
//                 ifEqFlags.push_back(true);
//               }

//               rewriter.setInsertionPoint(read);
//               auto value = rewriter.create<mlir::AffineLoadOp>(
//                   read.getLoc(), bufferArg, read.getMapOperands());

//               if (!ifExprs.empty()) {
//                 auto ifCondition = IntegerSet::get(dim, 0, ifExprs, ifEqFlags);
//                 auto ifOp = rewriter.create<AffineIfOp>(
//                     read.getLoc(), ifCondition, ifOperands, false);
//                 rewriter.setInsertionPointToStart(ifOp.getThenBlock());
//               }

//               rewriter.create<mlir::AffineStoreOp>(
//                   read.getLoc(), value, newBufferArg, read.getMapOperands());
//               continue;
//             }
//           }

//         // Otherwise, we need to create explicit data copy from the original
//         // buffer to new buffer if the new buffer is ever read.
//         if (!readUses.empty())
//           rewriter.create<memref::CopyOp>(loc, bufferArg, newBufferArg);
//       }
//     }
//     return success(hasChanged);
//   }
// };
// } // namespace

// namespace {
// struct MergeMultiProducer : public OpRewritePattern<ScheduleOp> {
//   using OpRewritePattern<ScheduleOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(ScheduleOp schedule,
//                                 PatternRewriter &rewriter) const override {
//     DominanceInfo DT(schedule);
//     bool hasChanged = false;

//     SmallVector<Value> externalBuffers;
//     for (auto arg : schedule.getBody().getArguments())
//       externalBuffers.push_back(arg);

//     for (auto buffer : externalBuffers) {
//       SmallVector<NodeOp> producers(getProducers(buffer));
//       if (producers.size() <= 1)
//         continue;

//       llvm::sort(producers, [&](NodeOp a, NodeOp b) {
//         return DT.properlyDominates(a, b);
//       });

//       auto allNodes = SmallVector<NodeOp>(schedule.getOps<NodeOp>().begin(),
//                                           schedule.getOps<NodeOp>().end());
//       auto ptr = llvm::find(allNodes, producers.front());
//       if (llvm::any_of(llvm::enumerate(producers), [&](auto node) {
//             return node.value() != *std::next(ptr, node.index());
//           }))
//         continue;

//       fuseNodeOps(producers, rewriter);
//       hasChanged = true;
//     }
//     return success(hasChanged);
//   }
// };
// } // namespace

// namespace {
// struct EliminateMultiProducer
//     : public EliminateMultiProducerBase<EliminateMultiProducer> {
//   void runOnOperation() override {
//     auto func = getOperation();
//     auto context = func.getContext();

//     mlir::RewritePatternSet patterns(context);
//     patterns.add<BufferMultiProducer>(context);
//     patterns.add<MergeMultiProducer>(context);
//     (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
//   }
// };
// } // namespace

// std::unique_ptr<Pass> scalehls::createEliminateMultiProducerPass() {
//   return std::make_unique<EliminateMultiProducer>();
// }
