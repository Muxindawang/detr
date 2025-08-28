import argparse
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
import models.transformer as transformer
from onnxsim import simplify
import onnx

import math


class BackboneBase(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        out = self.body(x)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation]
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, xs, mask):
        xs = self[0](xs)
        out = []
        pos = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(self[1](x, mask).to(x.dtype))
        return out, pos


class TransformerWrapper(nn.Module):
    """Transformer包装类，整合输入处理逻辑，减少外部输入节点"""

    def __init__(self, transformer, input_proj, query_embed, pos_embedder):
        super().__init__()
        self.transformer = transformer
        self.input_proj = input_proj  # 通道映射层
        self.query_embed = query_embed  # 可学习的查询嵌入（模型参数）
        self.pos_embedder = pos_embedder  # 位置编码器

    def forward(self, backbone_output):
        bs, _, h, w = backbone_output.shape
        mask = torch.ones(bs, h, w, dtype=torch.bool)

        src = self.input_proj(backbone_output)

        query_embed = self.query_embed.weight  # 形状: [num_queries, hidden_dim]

        bs, c, h, w = src.shape

        pos_embed = torch.ones(bs, c, h, w, dtype=torch.bool)

        return self.transformer(src, mask, query_embed, pos_embed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR模型组件转ONNX工具")
    parser.add_argument("--backbone", action="store_true", help="是否导出backbone")
    parser.add_argument("--pos", action="store_true", help="是否导出pos")
    parser.add_argument("--joiner", action="store_true", help="是否导出joiner")
    parser.add_argument(
        "--transformer", action="store_true", help="是否导出transformer"
    )

    parser.add_argument(
        "--dynamic_axes", action="store_true", help="是否导出动态形状的ONNX模型"
    )
    args = parser.parse_args()

    hidden_dim = 256
    nheads = 8
    dim_feedforward = 2048
    dec_layers = 6
    enc_layers = 6
    return_intermediate_dec = True

    batch_size = 1
    h = 800
    w = 800
    num_queries = 100

    img_input = torch.randn(batch_size, 3, h, w)

    backbone = Backbone(
        name="resnet50",
        train_backbone=False,
        return_interm_layers=False,
        dilation=False,
    )

    backbone.eval()
    with torch.no_grad():
        backbone_output = backbone(img_input)["0"]
        if args.backbone:
            torch.onnx.export(
                backbone,
                img_input,
                "backbone.onnx",
                input_names=["input"],
                output_names=["output"],
                export_params=True,
                training=False,
                opset_version=12,
                do_constant_folding=True,
            )
            backbone_onnx = onnx.load("backbone.onnx")
            onnx.checker.check_model(backbone_onnx)

            # test_input_shapes = {"input": img_input.shape}
            # model_simplified, check = simplify(
            #     backbone_onnx,
            #     test_input_shapes=test_input_shapes
            # )
            # assert check, "模型简化失败！"
            # onnx.save(model_simplified, "backbone_simplified.onnx")
            # print("简化后的模型已保存为: backbone_simplified.onnx")

            # onnx.checker.check_model(model_simplified)

    mask = torch.zeros(
        backbone_output.shape[0],
        backbone_output.shape[2],
        backbone_output.shape[3],
        dtype=torch.bool,
    )

    position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)
    position_embedding.eval()
    with torch.no_grad():
        pos_output = position_embedding(backbone_output, mask)
        if args.pos:
            torch.onnx.export(
                position_embedding,
                args=(backbone_output, mask),
                f="pos.onnx",
                input_names=["x", "mask"],
                output_names=["pos_embedding"],
                opset_version=12,
                do_constant_folding=True,
            )
            position_embedding_onnx = onnx.load("pos.onnx")
            onnx.checker.check_model(position_embedding_onnx)

            test_input_shapes = {"mask": mask.shape}
            model_simplified, check = simplify(
                position_embedding_onnx, test_input_shapes=test_input_shapes
            )
            assert check, "模型简化失败！"
            onnx.save(model_simplified, "pos_simplified.onnx")
            print("简化后的模型已保存为: pos_simplified.onnx")

            onnx.checker.check_model(model_simplified)

    joiner = Joiner(backbone, position_embedding)
    joiner.eval()
    if args.joiner:
        test_out, test_pos = joiner(img_input, mask)
        num_feats = len(test_out)

        torch.onnx.export(
            joiner,
            args=(img_input, mask),  # Joiner原有forward的两个输入：xs和mask
            f="joiner.onnx",
            input_names=["img_input", "mask"],  # 对应原有输入参数
            output_names=[f"output_feat_{i}" for i in range(num_feats)]
            + [f"output_pos_{i}" for i in range(num_feats)],
            opset_version=12,
            do_constant_folding=True,
        )

        # 验证原始模型
        original_model = onnx.load("joiner.onnx")
        onnx.checker.check_model(original_model)
        print("原始Joiner模型导出并验证成功")

        # torch.onnx.export(
        #         joiner,
        #         args=(img_input, mask),
        #         f="joiner.onnx",
        #         input_names=["input", "mask"],
        #         output_names=["img_input", "pos"],
        #         opset_version=12,
        #         do_constant_folding=True,
        #     )
        joiner_onnx = onnx.load("joiner.onnx")
        onnx.checker.check_model(joiner_onnx)

        test_input_shapes = {"img_input": img_input.shape, "mask": mask.shape}
        model_simplified, check = simplify(
            joiner_onnx, test_input_shapes=test_input_shapes
        )
        assert check, "模型简化失败！"
        onnx.save(model_simplified, "joiner_simplified.onnx")
        print("简化后的模型已保存为: joiner_simplified.onnx")

        onnx.checker.check_model(model_simplified)

    # 导出Transformer（优化部分）
    if args.transformer:
        # 1. 创建必要组件
        transformer_model = transformer.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            return_intermediate_dec=return_intermediate_dec,
        )
        input_proj = nn.Conv2d(
            backbone_output.shape[1], hidden_dim, kernel_size=1
        )  # 通道映射
        query_embed = nn.Embedding(
            num_queries, hidden_dim
        )  # 可学习的查询嵌入（作为模型参数）

        # 2. 包装Transformer，整合内部逻辑
        wrapped_transformer = TransformerWrapper(
            transformer=transformer_model,
            input_proj=input_proj,
            query_embed=query_embed,
            pos_embedder=position_embedding,
        )
        wrapped_transformer.eval()

        # 3. 导出ONNX
        # with torch.no_grad():
        #     # 导出模型（仅backbone_output作为输入）
        #     torch.onnx.export(
        #         wrapped_transformer,
        #         args=(backbone_output,),  # 仅需要backbone输出作为输入
        #         f="transformer.onnx",
        #         input_names=["backbone_output"],  # 输入节点仅保留backbone输出
        #         output_names=["hs", "memory"],
        #         opset_version=12,
        #         do_constant_folding=False
        #     )

        # 验证模型
        transformer_onnx = onnx.load("transformer.onnx")
        onnx.checker.check_model(transformer_onnx)

        input_shapes = {"backbone_output": backbone_output.shape}
        model_simplified, check = simplify(
            transformer_onnx, test_input_shapes=input_shapes
        )
        assert check, "模型简化失败！"

        # 4. 保存简化后的模型
        onnx.save(model_simplified, "transformer_simplified.onnx")
        print("简化后的模型已保存为: transformer_simplified.onnx")

        # 5. 验证简化后的模型
        onnx.checker.check_model(model_simplified)
        print("简化后的模型验证通过！")
