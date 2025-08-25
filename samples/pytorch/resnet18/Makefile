#!/bin/bash
BUDDY_BUILD_DIR := /home/dahaoming/buddy-mlir/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt

linalg:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" -o ./resnet18.mlir

clear:
	rm *.data *.mlir *.txt
