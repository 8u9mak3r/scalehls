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


enum class ReshapeKind {ExpandShape, Transpose, CollapseShape};

namespace {
  // Target shaped-type with redundant dimensions('1's) and element type in uniform with the src.
  // Thus, all the reshape operations themselves result in no redundant dims, regardless of those already in exist in old shapes
  mlir::ShapedType getDstShapedTypeUniformedWithSrc(mlir::ShapedType& srcShapedType, mlir::ShapedType& dstShapedType, ReshapeKind mode) {
    llvm::dbgs() << "Calling getDstShapedTypeUniformedWithSrc\n";

    auto srcShape = srcShapedType.getShape();
    auto dstShape = dstShapedType.getShape();
    llvm::SmallVector<int64_t> dstShapeUniformedWithSrc;

    size_t i = 0;
    size_t j = 0;
    if (mode == ReshapeKind::ExpandShape) {
      for (; i < srcShape.size(); i += 1) {
        if (srcShape[i] == 1) {
          dstShapeUniformedWithSrc.push_back(1);
          continue;
        }

        int64_t sum = 1;
        for (; j < dstShape.size(); j += 1) {
          if (sum == srcShape[i]) break;
          if (dstShape[j] == 1) continue;
          sum *= dstShape[j];
          dstShapeUniformedWithSrc.push_back(dstShape[j]);
        }
      }
    } else if (mode == ReshapeKind::Transpose) {
      for (; i < srcShape.size(); i += 1) {
        if (srcShape[i] == 1)  {
          dstShapeUniformedWithSrc.push_back(1);
          continue;
        }

        for (; j < dstShape.size(); j += 1) {
          if (dstShape[j] != 1) {
            dstShapeUniformedWithSrc.push_back(dstShape[j]);
            j += 1;
            break;
          }
        }
      }
    } else if (mode == ReshapeKind::CollapseShape) {
      for (; i < dstShape.size(); i += 1) {
        if (dstShape[i] == 1) continue;

        int64_t sum = dstShape[i];
        for (; j < srcShape.size(); j += 1) {
          if (srcShape[j] == 1) {
            dstShapeUniformedWithSrc.push_back(1);
            continue;
          }

          sum /= srcShape[j];

          if (sum == 1) {
            dstShapeUniformedWithSrc.push_back(dstShape[i]);
            j += 1;
            break;
          }
        }
      }
    }
    

    auto dstShapedTypeUniformedWithSrc = mlir::RankedTensorType::get(
      dstShapeUniformedWithSrc,
      srcShapedType.getElementType()
    );


    return dstShapedTypeUniformedWithSrc.cast<mlir::ShapedType>();
  }
} // namespace

namespace 
{
  mlir::AffineExpr getNewAffineExpr(mlir::AffineExpr expr, 
                                    llvm::ArrayRef<int64_t>& resultShape,
                                    llvm::SmallVector<mlir::ReassociationIndices, 4U>& reassociationIndices,
                                    llvm::SmallVector<int64_t, 4U>& staticLoopRanges,
                                    mlir::MLIRContext* ctx,
                                    ReshapeKind mode) {

    if (expr.isa<mlir::AffineConstantExpr>()) return expr;

    mlir::AffineExpr newExpr = mlir::getAffineConstantExpr(0, ctx);

    if (expr.isa<mlir::AffineDimExpr>()) {
      auto castExpr = expr.cast<AffineDimExpr>();
      auto loopRange = staticLoopRanges[castExpr.getPosition()];

      if (mode != ReshapeKind::CollapseShape) {
        assert(reassociationIndices.size() == staticLoopRanges.size());

        for (auto indices : reassociationIndices) {
          int64_t sum = 1;
          for (auto idx : indices) sum *= resultShape[idx];
          if (sum != loopRange) continue;

          for (auto idx : indices) {
            newExpr = newExpr + mlir::getAffineDimExpr(idx, ctx) * mlir::getAffineConstantExpr(sum / resultShape[idx], ctx);
            sum /= resultShape[idx];
          }
          break;
        }
      } else {
        assert(reassociationIndices.size() == resultShape.size());

        unsigned int idx = UINT32_MAX;
        for (auto indices : reassociationIndices) {
          idx += 1;
          auto it = llvm::find(indices, (int64_t)castExpr.getPosition());
          if (it == indices.end()) continue;

          newExpr = mlir::getAffineDimExpr(idx, ctx);
          for (auto _it = it + 1; _it < indices.end(); _it += 1) {
            newExpr = newExpr.floorDiv(mlir::getAffineConstantExpr(*_it, ctx));
          }
          newExpr = newExpr % mlir::getAffineConstantExpr(*it, ctx);
          break;
        }
      }
    }
    
    if (expr.isa<mlir::AffineBinaryOpExpr>()) {
      auto castExpr = expr.cast<AffineBinaryOpExpr>();
      auto lhs = castExpr.getLHS();
      auto rhs = castExpr.getRHS();
      auto kind = castExpr.getKind();

      auto newLHS = getNewAffineExpr(lhs, resultShape, reassociationIndices, staticLoopRanges, ctx, mode);
      auto newRHS = getNewAffineExpr(rhs, resultShape, reassociationIndices, staticLoopRanges, ctx, mode);

      switch (kind) {
        case AffineExprKind::Add:
          newExpr = newLHS + newRHS;
          break;
        case AffineExprKind::Mul:
          newExpr = newLHS * newRHS;
          break;
        case AffineExprKind::FloorDiv:
          newExpr = newLHS.floorDiv(newRHS);
          break;
        case AffineExprKind::CeilDiv:
          newExpr = newLHS.ceilDiv(newRHS);
          break;
        case AffineExprKind::Mod:
          newExpr = newLHS % newRHS;
          break;
        default:
          llvm_unreachable("Unsupported AffineExprKind");
      }
    }

    return newExpr;
  }
} // namespace 


namespace
{
  void modifyUsersFromExpandOp(llvm::SmallVector<mlir::Operation*, 128>& path, 
                               mlir::Operation* upliftedReshapeOp, 
                               PatternRewriter& rewriter,
                               ReshapeKind mode) {
    llvm::dbgs() << "Calling modifyUsersFromExpandOp\n";

    llvm::SmallVector<mlir::ReassociationIndices, 4U> reassociationIndices;
    switch (mode) {
      case ReshapeKind::ExpandShape:
        reassociationIndices = dyn_cast<tensor::ExpandShapeOp>(upliftedReshapeOp).getReassociationIndices();
        break;
    
      case ReshapeKind::Transpose: {
        int64_t rank = dyn_cast<linalg::TransposeOp>(upliftedReshapeOp).getPermutation().size();
        for (int64_t i = 0; i < rank; i += 1) {
          mlir::ReassociationIndices _ = {i};
          reassociationIndices.push_back(_);
        }
        break;
      }

      case ReshapeKind::CollapseShape:
        reassociationIndices = dyn_cast<tensor::CollapseShapeOp>(upliftedReshapeOp).getReassociationIndices();
        break;
    }
    
    auto* oldOp = *(path.end() - 1);
    auto* clonedOp = upliftedReshapeOp;
    auto* opToBeRebuiltAndErased = *(path.end() - 2);
    auto* iter = path.end() - 2;
    while (iter - path.begin() >= 0) {
      size_t operandIdxToBeReplaced = 0;
      for (auto operand : opToBeRebuiltAndErased->getOperands()) {
        if ((void*)operand.getDefiningOp() == (void*)oldOp) break;
        operandIdxToBeReplaced += 1;
      }

      if (iter == path.begin()) {
        switch (mode) {
          case ReshapeKind::ExpandShape:
            assert(dyn_cast<tensor::ExpandShapeOp>(opToBeRebuiltAndErased));
            break;

          case ReshapeKind::Transpose:
            assert(dyn_cast<linalg::TransposeOp>(opToBeRebuiltAndErased));
            break;

          case ReshapeKind::CollapseShape:
            assert(dyn_cast<tensor::CollapseShapeOp>(opToBeRebuiltAndErased));
            break;
        }

        if (clonedOp->getResult(0).getType() != opToBeRebuiltAndErased->getResult(0).getType()) {
          auto st0 = dyn_cast<mlir::ShapedType>(clonedOp->getResult(0).getType());
          auto st1 = dyn_cast<mlir::ShapedType>(opToBeRebuiltAndErased->getResult(0).getType());
          assert(st0.getShape().size() != st1.getShape().size() && !isReshaped(st0, st1));

          rewriter.setInsertionPoint(opToBeRebuiltAndErased);
          auto consistencyReshapeOp = st0.getShape().size() < st1.getShape().size() ? rewriter.create<tensor::ExpandShapeOp>(
              opToBeRebuiltAndErased->getLoc(),
              st1,
              clonedOp->getResult(0),
              mlir::getReassociationIndicesForReshape(st0, st1).value()
            ) : rewriter.create<tensor::CollapseShapeOp> (
              opToBeRebuiltAndErased->getLoc(),
              st1,
              clonedOp->getResult(0),
              mlir::getReassociationIndicesForCollapse(st0.getShape(), st1.getShape()).value()
            );

          clonedOp = consistencyReshapeOp;
        }
        
        for (auto& use : llvm::make_early_inc_range(opToBeRebuiltAndErased->getResult(0).getUses())) {
          if (isa<hls::YieldOp>(use.getOwner())) continue;
          use.set(clonedOp->getResult(0));
        }

        opToBeRebuiltAndErased->erase();
        llvm::dbgs() << "After Erase:\n";
        clonedOp->getParentOp()->dump();
        break;
      }

      if (isa<linalg::GenericOp>(opToBeRebuiltAndErased)) {
        auto oldGenericOp = dyn_cast<linalg::GenericOp>(opToBeRebuiltAndErased);
        auto loc = oldGenericOp.getLoc();  // location
        auto resultType = mlir::RankedTensorType::get(
          dyn_cast<mlir::ShapedType>(clonedOp->getResult(0).getType()).getShape(),
          dyn_cast<mlir::ShapedType>(oldGenericOp->getResult(0).getType()).getElementType()
        );  // result type
            
        llvm::SetVector<Value> inputs, outputs;
        // inputs
        size_t inputIdx = SIZE_MAX;
        for (auto input : oldGenericOp.getInputs()) {
          if (++inputIdx == operandIdxToBeReplaced) {
            inputs.insert(clonedOp->getResult(0));
            continue;
          }
          
          inputs.insert(input);
        }
        // outputs
        rewriter.setInsertionPoint(oldGenericOp);
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType()
        );
        outputs.insert(emptyOp.getResult());

        // indexing maps
        llvm::SmallVector<mlir::AffineMap, 8> affineMaps;
        llvm::SmallVector<mlir::AffineExpr, 8> affineExprs;
        mlir::MLIRContext* ctx = rewriter.getContext();

        auto oldIndexingMaps = oldGenericOp.getIndexingMapsArray();
        auto resultShape = resultType.getShape();
        auto staticLoopRanges = oldGenericOp.getStaticLoopRanges();
        size_t indexingMapIdx = SIZE_MAX;  // actually initiated as -1

        for (auto map : oldIndexingMaps) {
          indexingMapIdx += 1;

          if (indexingMapIdx == inputs.size()) {
            // indexing map for the output can always be an identity map
            affineMaps.push_back(mlir::AffineMap::getMultiDimIdentityMap(resultShape.size(), ctx));
            continue;
          } else if (indexingMapIdx == operandIdxToBeReplaced) {
            for (size_t i = 0; i < resultShape.size(); i += 1) {
              if (resultShape[i] == 1) affineExprs.push_back(mlir::getAffineConstantExpr(0, ctx));
              else affineExprs.push_back(mlir::getAffineDimExpr(i, ctx));
            }
            
            affineMaps.push_back(
              mlir::AffineMap::get(
                /*dimCount=*/resultShape.size(),
                /*symbolCount=*/0,
                affineExprs,
                ctx
              )
            );
            affineExprs.clear();
            continue;
          }

          /// Shape dimensions of the rest inputs are either exactly 1 or in correspondence with some combination
          /// of a certain slice of target shape's dimensions 
          /// e.g. with target shape being, say 16x12x64, other non-dominating inputs' shapes can be 1x768, 1x16 or similar

          for (auto expr : map.getResults()) {
            auto newExpr = getNewAffineExpr(expr, resultShape, reassociationIndices, staticLoopRanges, ctx, mode);
            affineExprs.push_back(newExpr);
          }

          auto newAffineMap = mlir::AffineMap::get(/*dimCount=*/resultShape.size(), /*symbolCount=*/0, affineExprs, ctx);
          affineMaps.push_back(newAffineMap);
          affineExprs.clear();
        }

        llvm::SmallVector<mlir::Attribute> affineMapAttrs;
        for (auto affineMap : affineMaps) affineMapAttrs.push_back(mlir::AffineMapAttr::get(affineMap));
        auto indexingMaps = rewriter.getArrayAttr(affineMapAttrs);
        affineMaps.clear();
        
        // iterator types
        llvm::SmallVector<mlir::Attribute> iteratorTypesAttrs;
        for (auto _ : resultShape) iteratorTypesAttrs.push_back(rewriter.getStringAttr("parallel"));
        auto iteratorTypes = rewriter.getArrayAttr(iteratorTypesAttrs);

        // new Generic with empty block args and ops
        rewriter.setInsertionPoint(oldGenericOp);
        auto newGenericOp = rewriter.create<linalg::GenericOp> (
          loc,
          TypeRange(resultType),
          ValueRange(inputs.getArrayRef()),
          ValueRange(outputs.getArrayRef()),
          indexingMaps,
          iteratorTypes,
          /*doc=*/nullptr, /*libraryCall=*/nullptr
        );

        // Add block args
        auto& newRegion = newGenericOp.getRegion();
        // OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&newRegion.emplaceBlock());

        auto newBlock = newRegion.begin();
        for (auto type : oldGenericOp.getBody()->getArgumentTypes()) {
          newBlock->addArgument(type, loc);
        }

        // clone block ops
        mlir::BlockAndValueMapping mapper;
        for (auto [oldArg, newArg] : llvm::zip(oldGenericOp.getBody()->getArguments(), newBlock->getArguments())) {
          mapper.map(oldArg, newArg);
        }

        for (auto& bodyOp : oldGenericOp.getBody()->getOperations()) {
          rewriter.clone(bodyOp, mapper);
        }

        newGenericOp.dump();
        newGenericOp.getOperation()->getParentOp()->dump();

        clonedOp = newGenericOp.getOperation();

      } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(opToBeRebuiltAndErased)) {
        assert(operandIdxToBeReplaced == 0);
        assert((void*)opToBeRebuiltAndErased->getOperand(0).getDefiningOp() == (void*)oldOp);

        auto oldReshapeOp = isa<tensor::ExpandShapeOp>(opToBeRebuiltAndErased) ?
            dyn_cast<tensor::ExpandShapeOp>(opToBeRebuiltAndErased) : dyn_cast<tensor::CollapseShapeOp>(opToBeRebuiltAndErased);
        auto oldSrcShapedType = dyn_cast<mlir::ShapedType>(oldReshapeOp->getOperand(0).getType());
        auto oldResultShapedType = dyn_cast<mlir::ShapedType>(oldReshapeOp->getResult(0).getType());

        // cannot be real reshapes, just add several redundant dimensions, in other words, add several '1's to the shape
        assert(!isReshaped(oldSrcShapedType, oldResultShapedType));

        auto newSrcShapedType = dyn_cast<mlir::ShapedType>(clonedOp->getResult(0).getType());
        auto oldSrcShape = oldSrcShapedType.getShape();
        auto newSrcShape = newSrcShapedType.getShape();
        auto oldResultShape = oldResultShapedType.getShape();
        llvm::SmallVector<int64_t> newResultShape;
        size_t reIndicesIdx = 0;
        
        if (mode != ReshapeKind::CollapseShape) {
          assert(reassociationIndices.size() == oldSrcShape.size());

          for (int64_t dim : oldResultShape) {
            // the old shape and the new one have exactly identical '1's in number and position
            if (dim == 1) {
              newResultShape.push_back(1);
              continue;
            }
          
            // the n-th non-'1' dim in the old shape is to be mapped to a set of new dims
            // in correspondence with the n-th reassociation index which isn't mapped to a single '1'
            for (; reIndicesIdx < reassociationIndices.size(); reIndicesIdx += 1) {
              auto indices = reassociationIndices[reIndicesIdx];
              if (!(indices.size() == 1 && newSrcShape[indices[0]] == 1)) {
                for (auto idx : indices) newResultShape.push_back(newSrcShape[idx]);
                reIndicesIdx += 1;
                break;
              }
            }
          }
        } else {
          // the else part share the same idea with the above
          assert(reassociationIndices.size() == newSrcShape.size());

          for (size_t i = 0; i < oldResultShape.size(); ) {
            if (oldResultShape[i] == 1) {
              newResultShape.push_back(1);
              i += 1;
              continue;
            }

            for (; reIndicesIdx < reassociationIndices.size(); reIndicesIdx += 1) {
              auto indices = reassociationIndices[reIndicesIdx];
              if (!(indices.size() == 1 && oldSrcShape[indices[0]] == 1)) {
                newResultShape.push_back(newSrcShape[reIndicesIdx]);
                reIndicesIdx += 1;
                i += indices.size();
                break;
              }
            }
          }
        }

        auto newResultShapedType = dyn_cast<mlir::ShapedType>(mlir::RankedTensorType::get(newResultShape, newSrcShapedType.getElementType()));
        rewriter.setInsertionPoint(oldReshapeOp);
        auto newReshapeOp = isa<tensor::ExpandShapeOp>(opToBeRebuiltAndErased) ? rewriter.create<tensor::ExpandShapeOp>(
            oldReshapeOp->getLoc(),
            newResultShapedType,
            clonedOp->getResult(0),
            mlir::getReassociationIndicesForReshape(newSrcShapedType, newResultShapedType).value()
          ) : rewriter.create<tensor::CollapseShapeOp>(
            oldReshapeOp->getLoc(),
            newResultShapedType,
            clonedOp->getResult(0),
            mlir::getReassociationIndicesForCollapse(newSrcShapedType.getShape(), newResultShapedType.getShape()).value()
          );

        switch (mode) {
          case ReshapeKind::ExpandShape:
            reassociationIndices = mlir::getReassociationIndicesForReshape(oldResultShapedType, newResultShapedType).value();
            break;

          case ReshapeKind::Transpose: {
            reassociationIndices.clear();

            // Reassociation indices of transpose can always be something like [[0], [1], [2], ..., [n]] 
            // --- an one-to-one map with no actual permutations embodied cuz the newShape has been permutated already
            auto ranks = newResultShapedType.getRank();
            for (int64_t i = 0; i < ranks; i += 1) {
              mlir::ReassociationIndices _ = {i};
              reassociationIndices.push_back(_);
            }
            break;
          }
        
          case ReshapeKind::CollapseShape:
            reassociationIndices = mlir::getReassociationIndicesForCollapse(
                                              oldResultShapedType.getShape(), newResultShapedType.getShape()).value();
            break;
        }

        newReshapeOp->dump();
        newReshapeOp->getParentOp()->dump();

        clonedOp = newReshapeOp;
      }

      oldOp = opToBeRebuiltAndErased;
      --iter;
      opToBeRebuiltAndErased = *iter;
    }
  }
} // namespace


namespace {
/// This pattern will outline ops with the specified type.
struct WalkOverTasksAndUpliftReshapeOps : public OpRewritePattern<hls::TaskOp> {
  using OpRewritePattern<hls::TaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hls::TaskOp op,
                                PatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::Operation*, 128> ops;
    llvm::SetVector<mlir::Operation*> opsToFuse;

    int defineUsePaths[128] = {-1, 0};
    int opsIdx = 0;

    llvm::dbgs() << "Calling WalkOverTasksAndUpliftReshapeOps\n";

    for (auto result : op.getResults()) {
      if (result.use_empty()) continue;
      llvm::dbgs() << "Current Result:\n";
      op.getYieldOp().getOperand(result.getResultNumber()).getDefiningOp()->dump();
      
      auto srcShapedType = dyn_cast<mlir::ShapedType>(result.getType());
      size_t resultIdx = result.getResultNumber();
      // result.getDefiningOp()->dump();
      ops.push_back(result.getDefiningOp());

      llvm::dbgs() << "Searching for all define use paths:\n";
      size_t _ = resultIdx;
      auto iter = ops.begin() - 1;

      // A Breadth-First Search(BFS) is performed here
      while ((++iter) != ops.end()) {
        (*iter)->dump();
        auto dstShapedType = dyn_cast<mlir::ShapedType>((*iter)->getResult(_).getType());
        if (isReshaped(srcShapedType, dstShapedType)) continue;

        for (auto& use : llvm::make_early_inc_range((*iter)->getResult(_).getUses())) {
          auto p = use.getOwner();
          if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, hls::YieldOp>(p) || isReductionTypeGenericOp(p)) continue;
          p->dump();
          ops.push_back(p);  
          defineUsePaths[++opsIdx] = iter - ops.begin();
        }
        // llvm::dbgs() << "\n";
        _ = 0;
      }

      llvm::SmallVector<mlir::Operation*, 128> path;
      for (size_t idx = opsIdx; idx != SIZE_MAX; idx -= 1) {
        llvm::dbgs() << ops.size() << "\n";
        iter = ops.begin() + idx;
        if (isa<hls::TaskOp>(*iter)) continue;
        auto dstShapedType = dyn_cast<mlir::ShapedType>((*iter)->getResult(0).getType());
        if (!isReshaped(srcShapedType, dstShapedType)) continue;
        llvm::dbgs() << "Walk through every path:\n";
        path.push_back(*iter);
        (*iter)->dump();
        while (iter != ops.begin()) {
          iter = ops.begin() + defineUsePaths[iter - ops.begin()];
          path.push_back(*iter);
          (*iter)->dump();
        }

        auto reshapeOp = *path.begin();
        auto taskOp =*(path.end() - 1);
        assert((void*)taskOp == (void*)op.getOperation());
        opsToFuse.insert(taskOp);  // SetVector is a set, so this insertion happens only once

        if (isa<tensor::ExpandShapeOp>(reshapeOp)) {
          // Target shaped-type with redundant dimensions and element type in uniform with the src.
          auto dstShapedTypeUniformedWithSrc = getDstShapedTypeUniformedWithSrc(srcShapedType, dstShapedType, ReshapeKind::ExpandShape);
        
          rewriter.setInsertionPointAfter(opsToFuse.back());
          auto upliftedExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
            opsToFuse.back()->getLoc(), 
            dstShapedTypeUniformedWithSrc, 
            taskOp->getResult(resultIdx), 
            mlir::getReassociationIndicesForReshape(srcShapedType, dstShapedTypeUniformedWithSrc).value()
          );

          llvm::dbgs() << "Uplifted ExpandShape:\n";
          upliftedExpandShapeOp.dump();
        
          modifyUsersFromExpandOp(path, upliftedExpandShapeOp.getOperation(), rewriter, ReshapeKind::ExpandShape);
          opsToFuse.insert(upliftedExpandShapeOp.getOperation());
        } else if (isa<linalg::TransposeOp>(reshapeOp)) {
          // Target shaped-type with redundant dimensions and element type in uniform with the src.
          auto dstShapedTypeUniformedWithSrc = getDstShapedTypeUniformedWithSrc(srcShapedType, dstShapedType, ReshapeKind::Transpose);
          

          rewriter.setInsertionPointAfter(opsToFuse.back());
          auto emptyOp = rewriter.create<tensor::EmptyOp>(
            opsToFuse.back()->getLoc(), dstShapedTypeUniformedWithSrc.getShape(), dstShapedTypeUniformedWithSrc.getElementType()
          );
          opsToFuse.insert(emptyOp.getOperation());

          rewriter.setInsertionPointAfter(opsToFuse.back());
          auto upliftedTransposeOp = rewriter.create<linalg::TransposeOp>(
            opsToFuse.back()->getLoc(),
            taskOp->getResult(resultIdx),
            emptyOp.getResult(),
            rewriter.getDenseI64ArrayAttr(computePermutation(srcShapedType, dstShapedTypeUniformedWithSrc))
          );

          llvm::dbgs() << "Uplifted Transpose:\n";
          emptyOp.dump();
          upliftedTransposeOp.dump();

          modifyUsersFromExpandOp(path, upliftedTransposeOp.getOperation(), rewriter, ReshapeKind::Transpose);
          opsToFuse.insert(upliftedTransposeOp.getOperation());

        } else if (isa<tensor::CollapseShapeOp>(reshapeOp)) {
          // Target shaped-type with redundant dimensions and element type in uniform with the src.
          auto dstShapedTypeUniformedWithSrc = getDstShapedTypeUniformedWithSrc(srcShapedType, dstShapedType, ReshapeKind::CollapseShape);
        
          rewriter.setInsertionPointAfter(opsToFuse.back());
          dstShapedTypeUniformedWithSrc.dump();
          auto upliftedCollapseShapeOp = rewriter.create<tensor::CollapseShapeOp>(
            opsToFuse.back()->getLoc(),
            dstShapedTypeUniformedWithSrc,
            taskOp->getResult(resultIdx),
            mlir::getReassociationIndicesForCollapse(srcShapedType.getShape(), dstShapedTypeUniformedWithSrc.getShape()).value()
          );

          llvm::dbgs() << "Uplifted CollapseShape:\n";
          upliftedCollapseShapeOp.dump();

          modifyUsersFromExpandOp(path, upliftedCollapseShapeOp.getOperation(), rewriter, ReshapeKind::CollapseShape);
          opsToFuse.insert(upliftedCollapseShapeOp.getOperation());
        } else {
          reshapeOp->emitError(
            "Oops! Reshape type operations aren't limited to ExpandShape, Transpose and CollapseShape.\n"
          );
          return failure();
        }
        path.clear();
      }

      llvm::dbgs() << "\n\n\n";
      ops.clear();
      opsIdx = 0;
    }

    if (opsToFuse.size() > 1) {
      llvm::dbgs() << "After TaskFusion:\n";

      // We assume that all operations behind the 2nd one are in fact replicates of the 2nd operation
      while (opsToFuse.size() > 2) {
        auto it = opsToFuse.begin() + 1;
        if (isa<tensor::EmptyOp>(*it)) it += 1;  // linalg.transpose always has an tensor.empty appended to it

        if (it == opsToFuse.end() - 1) break; // In a linalg.transpose situation the loop should terminate when there are three left
        assert(
          (*it)->getName() == opsToFuse.back()->getName() &&
          (*it)->getOperand(0).getType() == opsToFuse.back()->getOperand(0).getType() &&
          (*it)->getResult(0).getType() == opsToFuse.back()->getResult(0).getType()
        );
        opsToFuse.back()->getResult(0).replaceAllUsesWith((*it)->getResult(0));

        rewriter.eraseOp(opsToFuse.back());
        opsToFuse.pop_back();

        if (isa<linalg::TransposeOp>(*it)) {
          rewriter.eraseOp(opsToFuse.back());
          opsToFuse.pop_back();
        }
      }
      fuseOpsIntoTask(opsToFuse.getArrayRef(), rewriter, /*insertToLastOp=*/true).dump();
    }

    return success();
  }
};
} // namespace


namespace {
struct UpliftReshapesBetweenTasks
    : public UpliftReshapesBetweenTasksBase<UpliftReshapesBetweenTasks> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<WalkOverTasksAndUpliftReshapeOps>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), {false, true, 0L});
    patterns.clear();
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createUpliftReshapesBetweenTasksPass() {
  return std::make_unique<UpliftReshapesBetweenTasks>();
}
