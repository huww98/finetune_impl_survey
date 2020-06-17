# 关于实现细节对模型部分fine-tune行为的影响的调查

## 一、背景介绍

在迁移学习中，我们通常从已经经过训练的模型权重开始训练，这样的做法称为微调（fine-tune）。但使用机器学习框架实现这一操作的过程中，一些细节将会对模型的更新产生影响，特别是对batch normalization层的行为的影响。并最终对训练产生的模型造成影响。

### Batch Normalization的工作模式

工作模式的影响在两个方面：如何进行归一化，是否更新内部的统计量。

* 训练模式：BN层使用从当前batch中计算的统计量（均值与方差）对输入进行归一化，同时以指数移动平均的方式更新内部存储的统计量。
* 评估模式：BN层使用内部存储的统计量对输入进行归一化，且不更新内部存储的统计量。

## 二、实验目的

* 以PyTorch框架为例，调查不同的实现细节将对模型的微调行为产生怎样的影响。
* 调查TensorFlow 2.x中类似的细节。

## 三、实验方案设计

实验中所有代码均可在 https://github.com/huww98/finetune_impl_survey 查看。

### 模型

为模拟迁移学习中的常见场景，本实验的模型设计为两个部分：
* feature: 模拟已经经过预训练的特征提取器。它由全连接层，ReLU激活函数，BN层构成。
* classifier: 模拟分类器，通常是需要重点微调的部分。它由单层全连接层构成。

### 实验设置

改变不同的实现细节：
* 将feature中参数的requires_grad设为False（freeze）

  ```
  model.feature.requires_grad_(False)
  ```
  requires_grad设置将通知PyTorch是否需要跟踪该参数的计算，以便在反向传播中为它计算梯度。若设置为False，则在反向传播中不会为它计算梯度。进而在优化过程中不会改变该参数。

* 不将feature部分的参数传入optimizer中（not_optimize)

  optimizer只对传入其中的参数运行梯度优化算法，未传入的参数在优化过程中应该不会改变。

* 将feature部分设置为评估模式（eval_mode）

  ```
  model.feature.eval()
  ```

  将模型设置为评估模式将使BN层工作在评估模式，进而不更新内部存储的统计量。

### 结果评估方法

* 评估feature部分的参数（不包括BN中的统计量）微调中是否改变（params_changed）
* 评估BN中的统计量是否更新（bn_changed）

## 四、实验结果

<table>
    <thead>
        <tr>
            <td rowspan=2>#
            <td colspan=3>实验设置
            <td colspan=2>结果
        <tr>
            <td>freeze
            <td>not_optimize
            <td>eval_mode
            <td>params_changed
            <td>bn_changed
    </thead>
    <tbody>
        <tr><td>1<td>No <td>No <td>Yes<td>Yes<td>No
        <tr><td>2<td>Yes<td>No <td>Yes<td>No <td>No
        <tr><td>3<td>No <td>Yes<td>Yes<td>No <td>No
        <tr><td>4<td>Yes<td>Yes<td>Yes<td>No <td>No
        <tr><td>5<td>Yes<td>Yes<td>No <td>No <td>Yes
        <tr><td>6<td>No <td>No <td>No <td>Yes<td>Yes
    </tbody>
</table>

* 对比实验1-4可知：设置freeze和only_optimize的效果是类似的，其中任意一种都能达到不在优化过程中更新模型参数的效果，两者一起使用也能奏效。但单独使用only_optimize会造成不必要的内存和计算量开销，因为PyTorch依然为那些不需要更新的参数计算了梯度。

  但是，这两个设置都不能对BN层产生影响。在PyTorch中，BN中的统计量并不属于参数（parameters）而是属于buffer。因此，这些统计量不论如何设置都不会计算梯度，也不会传递给优化器。

* 对比实验4与5，1与6可知：eval_mode设置可固定BN中的统计量，不使用该设置则统计量会发生变化。且该设置的影响与前两个设置是正交的。

## 五、结果分析

PyTorch中不同的设置对模型的影响是明确的，且是与文档中描述一致的。当需要固定模型的一部分参数时，将`requires_grad`设置为False即可。当需要固定模型的一部分的BN统计量时，对该部分模型使用`model.eval()`即可。但需要注意，训练过程中通常会对整个模型调用`model.train()`，这会覆盖所有子模块中的模式设置。

但是，我尚未见到明确的结论：在微调过程中，若固定了一部分模型参数，这部分模型中的BN统计量是否需要同时固定。在开源的代码中，两种做法我都见过。

### TensorFlow API调查

本次调查中也对TensorFlow 2.x中的Keras API进行了调查，它与PyTorch是类似的。但也有以下特点：
* TensorFlow中每个层/每个模型都有一个`trainable`属性，该属性同时承担了PyTorch中的`requires_grad`和`model.eval()`的功能。即，将`trainable`设置为False后，将不会为该部分参数计算梯度，同时将BN层置于评估模式。
* TensorFlow中模型有两种运行模式。不同模式将对BN的运行模式造成影响。
    * 训练模式：在`model.fit()`、`model(x, training=True)`等情况下使用。此时BN层的模式遵守`trainable`属性的设置。
    * 评估模式：在`model.predict()`、`model.evaluate()`、`model(x, training=False)`等情况下使用。此时BN层无论其`trainable`属性的设置如何，均工作在评估模式下。
* TensorFlow只将计算了梯度的参数传递给optimizer，`trainable`设置为False的参数将始终不会传递给optimizer。
* TensorFlow中，BN层的统计量是特殊的参数，其`trainable`始终设置为False，且无法通过公开API将其更改为True。这一定程度上保证了这些统计量不会参与到梯度计算和optimizer的工作中。（作为对比，PyTorch中的统计量是buffer而不是参数，同样能保证其不参与到梯度计算和optimizer工作中。）
