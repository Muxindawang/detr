#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#                ~~~Medcare AI Lab~~~

# 导入必要的库
import io  # 用于内存中的字节流操作
import sys  # 系统相关操作
import argparse  # 命令行参数解析

import numpy as np  # 数值计算库
import onnx  # ONNX模型操作库
import onnxruntime  # ONNX模型推理引擎
from onnxsim import simplify  # ONNX模型简化工具
import onnx_graphsurgeon as gs  # ONNX图结构操作工具

import torch  # PyTorch深度学习框架
from hubconf import detr_resnet50  # 导入DETR模型（来自自定义配置）


class ONNXExporter:
    """ONNX模型导出工具类，用于将PyTorch模型导出为ONNX格式并进行验证和处理"""

    @classmethod
    def setUpClass(cls):
        """类初始化方法，设置随机种子以保证结果可复现"""
        torch.manual_seed(123)

    def run_model(self, model, onnx_path, inputs_list, dynamic_axes=False, tolerate_small_mismatch=False, 
                  do_constant_folding=False, output_names=None, input_names=None):
        """
        将PyTorch模型导出为ONNX格式
        
        参数:
            model: 待导出的PyTorch模型
            onnx_path: ONNX模型保存路径
            inputs_list: 测试输入数据列表
            dynamic_axes: 是否导出动态形状的ONNX模型
            tolerate_small_mismatch: 是否容忍PyTorch与ONNX推理结果的微小差异
            do_constant_folding: 是否启用常量折叠优化
            output_names: 输出节点名称列表
            input_names: 输入节点名称列表
        """
        model.eval()  # 设置模型为评估模式

        # 创建内存字节流用于临时存储ONNX模型
        onnx_io = io.BytesIO()
        
        # 导出模型到内存和指定路径
        torch.onnx.export(model, inputs_list[0], onnx_io,
            input_names=input_names, output_names=output_names,
            export_params=True, training=False, opset_version=12, do_constant_folding=do_constant_folding)
        torch.onnx.export(model, inputs_list[0], onnx_path,
            input_names=input_names, output_names=output_names,
            export_params=True, training=False, opset_version=12, do_constant_folding=do_constant_folding)
        
        print(f"[INFO] ONNX模型导出成功！保存路径: {onnx_path}")

        # 使用ONNX Runtime验证导出的模型
        for test_inputs in inputs_list:
            with torch.no_grad():  # 禁用梯度计算，提高效率
                # 处理输入格式
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                # 获取PyTorch模型输出
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            # 验证ONNX模型输出与PyTorch是否一致
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

        print("[INFO] 使用ONNX Runtime验证导出模型成功！")

        # 导出动态形状的ONNX模型（如果需要）
        if dynamic_axes:
            torch.onnx.export(model, inputs_list[0], './detr_dynamic.onnx', 
                dynamic_axes={
                    input_names[0]: {0: '-1'},  # 输入第0维动态
                    output_names[0]: {0: '-1'}, # 输出1第0维动态
                    output_names[1]: {0: '-1'}  # 输出2第0维动态
                },
                input_names=input_names, output_names=output_names, 
                verbose=True, opset_version=12)
            
            print(f"[INFO] 动态形状ONNX模型导出成功！动态形状: {dynamic_axes} 保存路径: ./detr_dynamic.onnx")

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):
        """
        使用ONNX Runtime验证导出的模型
        
        参数:
            onnx_io: 内存中的ONNX模型字节流
            inputs: PyTorch模型的输入
            outputs: PyTorch模型的输出
            tolerate_small_mismatch: 是否容忍微小差异
        """
        # 展平输入输出（处理嵌套结构）
        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        # 将PyTorch张量转换为NumPy数组
        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        # 创建ONNX Runtime推理会话
        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # 构造ONNX输入
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        # 执行ONNX模型推理
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 比较PyTorch与ONNX的输出差异
        for i in range(0, len(outputs)):
            try:
                # 严格校验（容忍一定范围内的误差）
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    # 容忍微小差异时检查错误信息
                    if "(0.00%)" not in str(error):
                        raise  # 如果不是0.00%的差异则抛出异常
                else:
                    raise  # 不容忍差异则抛出异常

    @staticmethod
    def check_onnx(onnx_path):
        """
        检查ONNX模型的有效性
        
        参数:
            onnx_path: ONNX模型路径
        """
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)  # 执行ONNX模型检查
        print(f"[INFO]  ONNX模型: {onnx_path} 检查成功!")


    @staticmethod
    def onnx_change(onnx_path):
        '''
        修改ONNX模型中的特定节点，解决TensorRT推理全为0的问题（导师提供的修复代码）
        
        参数:
            onnx_path: 待修改的ONNX模型路径
        '''
        # 不同批次大小对应的节点配置
        node_configs = [(2682,2684),(2775,2777),(2961,2963),(3333,3335),(4077,4079)]
        if 'batch_2' in onnx_path:
            node_number = node_configs[1]
        elif 'batch_4' in onnx_path:
            node_number = node_configs[2]
        elif 'batch_8' in onnx_path:
            node_number = node_configs[3]
        elif 'batch_16' in onnx_path:
            node_number = node_configs[4]
        else:
            node_number = node_configs[0]  # 默认配置（batch_size=1）

        # 加载并修改ONNX模型
        graph = gs.import_onnx(onnx.load(onnx_path))
        for node in graph.nodes:
            # 修改指定Gather节点的输入值
            if node.name == f"Gather_{node_number[0]}":
                print(node.inputs[1])
                node.inputs[1].values = np.int64(5)  # 修改为固定值5
                print(node.inputs[1])
            elif node.name == f"Gather_{node_number[1]}":
                print(node.inputs[1])
                node.inputs[1].values = np.int64(5)  # 修改为固定值5
                print(node.inputs[1])
                
        # 保存修改后的模型
        onnx.save(gs.export_onnx(graph), onnx_path)
        print(f"[INFO] ONNX修改完成，保存在{onnx_path}.")


if __name__ == '__main__':
    """主函数：解析命令行参数并执行模型导出流程"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DETR模型转ONNX模型工具')
    parser.add_argument('--model_dir', type=str, default='./detr-r50-e632da11.pth', 
                        help='DETR PyTorch模型保存路径')    
    parser.add_argument('--dynamic_axes', action="store_true", 
                        help='是否导出动态形状的ONNX模型')
    parser.add_argument('--check', action="store_true", 
                        help='是否检查ONNX模型有效性')
    parser.add_argument('--onnx_dir', type=str, default="./detr.onnx", 
                        help="ONNX模型保存路径")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="输入批次大小")

    args = parser.parse_args()  # 解析参数

    # 加载PyTorch模型
    # 注意：num_classes=20+1 表示20个类别+1个背景类
    detr = detr_resnet50(pretrained=False, num_classes=90+1).eval()
    # 加载模型权重
    state_dict = torch.load(args.model_dir, map_location='cuda')  # 加载权重文件
    detr.load_state_dict(state_dict["model"])  # 加载模型参数
    
    # 创建测试输入（dummy数据）
    dummy_image = [torch.ones(args.batch_size, 3, 800, 800)]  # 形状：[batch_size, 3, 800, 800]

    # 执行ONNX导出
    onnx_export = ONNXExporter()
    onnx_export.run_model(
        detr, 
        args.onnx_dir, 
        dummy_image,
        input_names=['inputs'],  # 输入节点名称
        dynamic_axes=args.dynamic_axes,
        output_names=["pred_logits", "pred_boxes"],  # 输出节点名称（分类logits和边界框）
        tolerate_small_mismatch=True
    )

    # 检查ONNX模型（如果指定）
    if args.check:
        ONNXExporter.check_onnx(args.onnx_dir)

    # 简化ONNX模型
    print('[INFO] 正在简化模型...')
    model = onnx.load(args.onnx_dir)
    # 执行模型简化
    simplified_model, check = simplify(
        model,
        input_shapes={'inputs': [args.batch_size, 3, 800, 800]},  # 输入形状
        dynamic_input_shape=args.dynamic_axes  # 是否动态输入形状
    )

    # 保存简化后的模型
    simplified_onnx_path = args.onnx_dir[:-5] + "_sim.onnx"  # 在原文件名后添加"_sim"
    onnx.save(simplified_model, simplified_onnx_path)

    # 修改简化后的ONNX模型（解决TRT推理问题）
    onnx_export.onnx_change(simplified_onnx_path)

    # 以下为手动执行模型简化的命令（备用）
    # $ python3 -m onnxsim detr.onnx detr_sim.onnx
    # $ python3 -m onnxsim detr_dynamic.onnx detr_dynamic_sim.onnx