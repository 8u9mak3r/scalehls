// //===----------------------------------------------------------------------===//
// //
// // Copyright 2020-2021 The ScaleHLS Authors.
// //
// //===----------------------------------------------------------------------===//

// #include "mlir/IR/Dominance.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "scalehls/Transforms/Passes.h"
// #include "scalehls/Transforms/Utils.h"

// using namespace mlir;
// using namespace scalehls;
// using namespace hls;

// static void rebuildNodeOpRecursively(hls::NodeOp& node,
//                                     SmallVector<Value> oldBuffers,
//                                     SmallVector<Value> newBuffers,
//                                     PatternRewriter& rewriter) {
//   assert((
//       oldBuffers.begin()->getType() == newBuffers.begin()->getType() &&
//       llvm::all_of(oldBuffers, [&](Value& v) { return v.getType() == oldBuffers.begin()->getType(); }) &&
//       llvm::all_of(newBuffers, [&](Value& v) { return v.getType() == newBuffers.begin()->getType(); })
//     ) && "Type Mismatch between or within old buffers and new buffers."
//   );

//   auto oldBufferIndices = SmallVector<unsigned>();
//   llvm::transform(oldBuffers, std::back_inserter(oldBufferIndices),
//                   [&](Value& buffer) { return llvm::find(node.getOutputs(), buffer) - node.getOutputs().begin() + node.getNumInputs(); });
//   unsigned numInputsAndOutputs = node.getNumInputs() + node.getNumOutputs();
//   assert(llvm::all_of(oldBufferIndices, [&](unsigned idx) { return idx < numInputsAndOutputs; }));
  
//   auto& nodeBody = node.getBody();
//   auto oldBufferArgs = SmallVector<Value>(), newBufferArgs = SmallVector<Value>();
//   llvm::transform(oldBufferIndices, std::back_inserter(oldBufferArgs),
//                   [&](unsigned idx) { return nodeBody.getArgument(idx); } );
//   llvm::transform(newBuffers, std::back_inserter(newBufferArgs),
//                   [&](Value& v) { return nodeBody.insertArgument(numInputsAndOutputs, v.getType(), v.getLoc()); });

//   auto usesVec = SmallVector<mlir::Value::use_range>();
//   auto users = SmallVector<Operation*>();
//   llvm::transform(oldBufferArgs, std::back_inserter(usesVec), [&](Value& v) { return v.getUses(); });
//   assert(llvm::all_of(usesVec, [](mlir::Value::use_range& uses) { return llvm::hasSingleElement(uses); }) 
//                                 && "Multiple uses on output buffer within a single node are not allowed.");
//   llvm::transform(usesVec, std::back_inserter(users),
//                   [](mlir::Value::use_range& uses) { return uses.begin()->getOwner(); });
//   assert(llvm::all_equal(users) || llvm::all_of(users, [](Operation* user) { return isa<AffineStoreOp>(user); }));
//   auto user = *users.begin();

//   if (isa<hls::ScheduleOp>(user)) {
//     auto schedule = dyn_cast<hls::ScheduleOp>(user);
//     auto& scheduleBody = schedule.getBody();
//     auto oldBufferIdxInSchedule = SmallVector<unsigned>();
//     llvm::transform(usesVec, std::back_inserter(oldBufferIdxInSchedule),
//                     [](mlir::Value::use_range& uses) { return uses.begin()->getOperandNumber(); } );

//     auto oldBufferArgsInSchedule = SmallVector<Value>(), newBufferArgsInSchedule = SmallVector<Value>();
//     llvm::transform(oldBufferIdxInSchedule, std::back_inserter(oldBufferArgsInSchedule),
//                     [&](unsigned idx) { return scheduleBody.getArgument(idx); });
//     llvm::transform(newBufferArgs, std::back_inserter(newBufferArgsInSchedule),
//                   [&](Value& v) { return scheduleBody.insertArgument(scheduleBody.getNumArguments(), v.getType(), v.getLoc()); });

//     SmallVector<Value> scheduleInputs = schedule.getOperands();
//     scheduleInputs.append(newBufferArgs);
//     assert((
//         scheduleInputs.size() == scheduleBody.getNumArguments() &&
//         llvm::all_of(
//           llvm::zip(scheduleInputs, scheduleBody.getArguments()),
//           [&](std::tuple<mlir::Value&, mlir::BlockArgument&> pair) { return std::get<0>(pair).getType() == std::get<1>(pair).getType(); }
//         )
//       ) && "Mismatch between practical inputs and block arguments in ScheduleOp."
//     );

//     auto usesVecInSchedule = SmallVector<mlir::Value::use_range>();
//     auto usersInSchedule = SmallVector<Operation*>();
//     llvm::transform(oldBufferArgsInSchedule, std::back_inserter(usesVecInSchedule), [](Value& v) { return v.getUses(); });
//     // The first use in the useslist is bound to be the node writes data to the buffer, for now
//     // FIXME: A more general method to locate that node
//     llvm::transform(usesVecInSchedule, std::back_inserter(usersInSchedule),
//                     [](mlir::Value::use_range& uses) { return uses.begin()->getOwner(); } );
//     llvm::all_equal(usersInSchedule);
//     auto userInSchedule = *usersInSchedule.begin();
//     assert(isa<hls::NodeOp>(userInSchedule));

//     auto nodeOpInSchedule = dyn_cast<hls::NodeOp>(userInSchedule);
//     rebuildNodeOpRecursively(nodeOpInSchedule, oldBufferArgsInSchedule,
//                             newBufferArgsInSchedule, rewriter);
    
//     rewriter.setInsertionPoint(schedule);
//     auto newSchedule = rewriter.create<hls::ScheduleOp>(schedule.getLoc(), scheduleInputs);
//     rewriter.inlineRegionBefore(scheduleBody, newSchedule.getBody(), newSchedule.getBody().begin());

//     rewriter.eraseOp(schedule);
//   } else if (isa<AffineStoreOp>(user)) {
//     auto producersVec = SmallVector<SmallVector<hls::NodeOp>>();
//     llvm::transform(oldBuffers, std::back_inserter(producersVec),
//                     [](Value& buffer) { return getProducers(buffer); });
//     assert(llvm::all_of(oldBuffers, [](Value& buffer) { return getConsumers(buffer).size() <= 1; } )
//             && "Multiple consumers on the output buffer detected in the innermost layer of the node.");
//     assert(llvm::all_of(producersVec, 
//             [&](SmallVector<hls::NodeOp>& producers) { return llvm::hasSingleElement(producers) && *producers.begin() == node; })
//             && "Multiple producers using the output buffer as input detected in the innermost layer of the node.");

//     auto store = dyn_cast<AffineStoreOp>(user).getOperation();

//     BlockAndValueMapping mapper;
//     rewriter.setInsertionPoint(store);
//     for (auto arg : newBufferArgs) {
//       // Types of all these old and new buffer args are all the same anyway
//       mapper.map(*oldBufferArgs.begin(), arg);
//       rewriter.clone(*store, mapper);
//       mapper.clear();
//     }
//   } else {
//     llvm_unreachable("An abnormal operations other than hls.schedule and affine.store detected, which is not supported currently.");
//   }

//   rewriter.setInsertionPoint(node);
//   SmallVector<Value> outputs = node.getOutputs();
//   outputs.append(newBuffers);
//   auto newNode = rewriter.create<hls::NodeOp>(
//     node.getLoc(),
//     node.getInputs(),
//     outputs,
//     node.getParams(),
//     node.getInputTapsAttr(),
//     node.getLevelAttr()
//   );
//   rewriter.inlineRegionBefore(nodeBody, newNode.getBody(), newNode.getBody().begin());
//   rewriter.eraseOp(node);
// }

// /// This pass currently supports one and only one code pattern:
// /// hls.node (...) -> (%0) { # %0 as output buffer to be written into later
// ///   hls.schedule (......) {
// ///     hls.node (...) -> (...) {} # read inputs
// ///     hls.node (...) -> (...) {} # read inputs
// ///     hls.node (...) -> (...) {} # read inputs
// ///     ......
// ///     hls.node (%0) -> (...) {} # read outputs as initial value
// ///     hls.node (...) -> (...) {} # computation
// ///     hls.node (...) -> (%0) {# write outputs
// ///       affine.load ...
// ///       affine store ..., %0
// ///     }
// ///   }
// /// }
// /// All the others may cause the whole pass pipeline's crash immediately, as you can see many asserts
// /// and even one abort() in the source code up above
// namespace {
// struct InsertExtraStoreOps : public OpRewritePattern<NodeOp> {
//   using OpRewritePattern<NodeOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(NodeOp node,
//                                 PatternRewriter &rewriter) const override {
//     auto loc = rewriter.getUnknownLoc();
//     SmallVector<Value> oldBuffers, newBuffers;

//     auto hasChanged = false;
//     for (auto output : node.getOutputs()) {
//       // DRAM buffer is not considered - the dependencies associated with them
//       // are handled later by tokens.
//       // if (isExternalBuffer(output))
//       //   continue;

//       auto consumers = getDependentConsumers(output, node);
//       if (consumers.size() < 2)
//         continue;

//       hasChanged = true;
//       rewriter.setInsertionPoint(node);
      
//       // Insert a buffer for each consumer.
//       auto outputBufferOp = output.getDefiningOp<hls::BufferOp>();
//       for (auto pair : llvm::enumerate(consumers)) {
//         if (pair.index() == 0) continue;

//         auto consumer = pair.value();
//         auto buffer = rewriter.create<BufferOp>(loc, output.getType(), outputBufferOp.getDepthAttr(), outputBufferOp.getInitValueAttr());
//         output.replaceUsesWithIf(
//             buffer, [&](OpOperand &use) { return use.getOwner() == consumer; });
//         newBuffers.push_back(buffer);
//       }
//       oldBuffers.push_back(output);
//     }
//     if (hasChanged) rebuildNodeOpRecursively(node, oldBuffers, newBuffers, rewriter);
//     return success(hasChanged);
//   }
// };
// } // namespace

// namespace {
// struct EliminateMultiConsumer
//     : public EliminateMultiConsumerBase<EliminateMultiConsumer> {
//   void runOnOperation() override {
//     auto func = getOperation();
//     auto context = func.getContext();

//     mlir::RewritePatternSet patterns(context);
//     patterns.add<InsertExtraStoreOps>(context);
//     (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
//   }
// };
// } // namespace

// std::unique_ptr<Pass> scalehls::createEliminateMultiConsumerPass() {
//   return std::make_unique<EliminateMultiConsumer>();
// }


//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace {
struct InsertForkNode : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    auto loc = rewriter.getUnknownLoc();

    auto hasChanged = false;
    for (auto output : node.getOutputs()) {
      // DRAM buffer is not considered - the dependencies associated with them
      // are handled later by tokens.
      if (isExternalBuffer(output))
        continue;

      auto consumers = getDependentConsumers(output, node);
      if (consumers.size() < 2)
        continue;

      hasChanged = true;
      rewriter.setInsertionPointAfter(node);
      SmallVector<Value> buffers;
      SmallVector<Location> bufferLocs;

      // Insert a buffer for each consumer.
      for (auto consumer : consumers) {
        auto buffer = rewriter.create<BufferOp>(loc, output.getType());
        output.replaceUsesWithIf(
            buffer, [&](OpOperand &use) { return use.getOwner() == consumer; });
        buffers.push_back(buffer);
        bufferLocs.push_back(loc);
      }

      // Create a new fork node.
      auto fork = rewriter.create<NodeOp>(loc, output, buffers);
      auto block = rewriter.createBlock(&fork.getBody());
      auto outputArg = block->addArgument(output.getType(), output.getLoc());
      auto bufferArgs = block->addArguments(ValueRange(buffers), bufferLocs);

      // Create explicit copy from the original output to the buffers.
      rewriter.setInsertionPointToStart(block);
      for (auto bufferArg : bufferArgs)
        rewriter.create<memref::CopyOp>(loc, outputArg, bufferArg);

      fork->setAttr("Just Make Copies", rewriter.getBoolAttr(true));
    }
    return success(hasChanged);
  }
};
} // namespace

namespace {
struct EliminateMultiConsumer
    : public EliminateMultiConsumerBase<EliminateMultiConsumer> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<InsertForkNode>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createEliminateMultiConsumerPass() {
  return std::make_unique<EliminateMultiConsumer>();
}

