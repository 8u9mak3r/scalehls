'''ResNet18 in PyTorch.
Modified based on (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
'''

import os
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._decomp import remove_decompositions

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ResNet18 model AOT importer")
parser.add_argument(
    "--output-dir", 
    type=str, 
    default="./", 
    help="Directory to save output files."
)
args = parser.parse_args()

# Ensure output directory exists
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]).eval()

# import torch_mlir
# module =  torch_mlir.compile(ResNet18(), torch.ones(1, 3, 32, 32), output_type=torch_mlir.OutputType.TOSA)
# print(module)
# exit(0)

DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.max_pool2d_with_indices.default,
]

remove_decompositions(inductor_decomp, DEFAULT_DECOMPOSITIONS)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

data = torch.randn([1, 3, 32, 32])

with torch.no_grad():
    graphs = dynamo_compiler.importer(ResNet18(), data)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]
current_path = os.path.dirname(os.path.abspath(__file__))

float32_param = np.concatenate(
    [
        param.detach().numpy().reshape([-1]) for param in params
    ]
)
float32_param.tofile(Path(output_dir) / "arg0.data")
