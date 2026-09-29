# 多模态学习笔记（示例语料）

> 用途：给 LLM 抽取管线试跑用的示例笔记。
> 运行：python extract.py data/corpus/demo_notes.md --dry-run

## Vision Transformer

ViT 把图像切成 16x16 的 patch，每个 patch 过线性投影变成 token 序列，
再套标准 Transformer 编码器做分类。它证明了纯注意力架构不依赖卷积也能在
图像分类上打平 CNN，前提是预训练数据足够大（JFT-300M 量级）。
ViT 的前置知识就是 Transformer 和 embedding；相比 CNN 它缺少归纳偏置，
小数据集上容易过拟合。

## CLIP

CLIP 用对比学习对齐图文两种模态：图像编码器（ViT 或 ResNet）和文本编码器
（Transformer）把各自输入映射到同一向量空间，4 亿图文对上做 InfoNCE 损失，
拉近配对样本、推远非配对样本。训练好的 CLIP 可以做零样本分类：
把类别名写成 "a photo of a dog" 之类文本，比较图像向量和各类文本向量的余弦相似度。
它也是文生图（Stable Diffusion 等）里文本条件注入的基础组件。

## 扩散模型

扩散模型的思想是：前向过程逐步给图像加高斯噪声直到变成纯噪声，
反向过程学一个 U-Net 去噪网络把纯噪声一步步还原回图像。
采样时从随机噪声出发迭代去噪。DDPM 奠定了框架，DDIM 把采样加速到几十步。
Stable Diffusion 把去噪搬到 VAE 的隐空间里做，并用 CLIP 文本编码器做条件
（cross-attention 注入），大幅降低算力门槛，这也叫潜在扩散模型 LDM。
理解它需要先掌握 VAE 和注意力机制。CFG（classifier-free guidance）通过
有条件和无条件预测的线性外推控制生成与文本的一致性，是实际部署的标配技巧。

## 复习记录

今天把 ViT 的 patch embedding 手推了一遍；CLIP 的对比损失矩阵化计算
（2N x N 相似度矩阵）之前实现过一次。扩散模型的采样公式还有点模糊，
下周找 DDPM 论文的伪代码对照推导。
