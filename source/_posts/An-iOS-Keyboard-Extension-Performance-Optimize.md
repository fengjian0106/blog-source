title: iOS输入法性能优化  
date: 2014-09-26 10:25:52
tags: [iOS, keyborad, LevelDB, AutoLayout]
---

一年前指导同事开发了一款andriod版的输入法(非中文)，其中的词库引擎，是我做的技术选型并且在iOS平台上做了原型验证，采用的是 `Ternary Search Tree`。一年后，也就是最近，随着iOS8的发布，我们也要发布一款iOS版的输入法。

### 遇到的主要问题
1. `Ternary Search Tree` 实现 `prefix match` 的速度很快，但是因为只是一个纯粹的内存数据结构，所以输入法词库的容量是一个瓶颈。在android平台上，考虑多方面因素后，我们的词库中只有3万左右的单词量。iOS平台上做原型验证的时候，词库容量也只能做到10万左右。但是实际的业务需求中，是希望词库容量可以进一步增大的。
2. iOS版本输入法的开发过程中，遇到了另外一个问题，就是键盘页面的加载速度和切换速度有点慢，用户能够感觉出来。

### 解决办法
#### 词库容量
词库容量扩充这个问题，其实一直是一个难题，在 `Ternary Search Tree` 上也做过一些优化，但是变化并不明显。反而是在做server端开发，学习 `LevelDB` 的时候，碰巧发现 `LevelDB` 是一个很好的替代。首先 `LevelDB` 支持 `prefix search`，而且搜索速度也很快，测试数据表明完全满足我们的业务需求，其次 `LevelDB` 是将数据存储在文件系统上的，没有了内存大小的限制，词库的容量很轻松就可以扩充100倍以上，而且有了这种近乎无限的词库容量后，之前有一些需要复杂算法甚至很难实现的业务需求，现在也可以在超大词库的基础上，用**“简单但是粗暴”**的算法实现出来。

iOS平台上的 `Core Data` 是一套相当好用的数据持久化存储框架，唯一*`可能存在`*的问题就是性能，因此有些开发者在某些场景中，还是愿意去选择使用 `SQLite` 。有了这次的开发经验后，相信在某些应用场景中， `LevelDB` 也将会是一个很好的替代方案，比如 Square 开源的 [Viewfinder](https://github.com/viewfinderco/viewfinder.git) 中的客户端，就是用  `LevelDB` 实现的数据存储。`LevelDB` 的核心是 `LSM-Tree`，其实 [SQLite4](http://sqlite.org/src4/doc/trunk/www/lsmusr.wiki) 的核心，也是 `LSM-Tree`，小伙伴们，你们知道吗 :-)

#### 页面加载和切换速度
说实话，页面加载速度这个问题，挺出乎意料的，以前我们团队也做了这么多iOS应用了，从来没有在页面速度上遇到过问题，用 `Instruments`、 `NSLog` 对比分析了一遍，测量出来的页面加载时间，也和其他应用中页面加载消耗的时间差不多。大家讨论了一下为什么用户会觉得慢，得出的结论是，输入法本来就是一个效率型的工具app，用户心理的期待之一，就是键盘的速度要快，而普通类型的app，用户对速度不会这么敏感。

问题已经出来了，还是得想办法去优化，吭哧吭哧写代码调试，从3个方面压缩了页面加载切换时消耗的时间：

1. 键盘的view，是分了好几个层次的，当作为container的UIView加载完成后，就立刻让键盘先显示出来，然后再触发加载真正的keyboard view，这样给用户的一个心理感觉就是键盘弹出的速度很快。
2. 键盘切换的时候，不再每次都重新从xib中加载对应的view，而是将view缓存在cache里面，用空间换时间。
3. 移除了keyboard view中每个key view上的 `Auto Layout` 约束条件，直接在 `layoutSubviews` 方法中设置subview的 `frame`，关于这个优化思路，可以看看 [Optimising Autolayout](http://pilky.me/36/)。需要强调的是，我们并不是否定 `Auto Layout` ，实际上我们团队现在采用的思路是 `Auto Layout` 和 `Manual Frame Layout` 一起使用，代码布局和xib布局一起使用，根据页面的需求做出更合适的选择。
4. 2014-10-16更新，借助Facebook出品的神器https://github.com/facebook/AsyncDisplayKit.git，又抠了一些性能出来:]

这款输入法app，我们还全面切换到使用 `ReactiveCocoa` 这个框架进行开发，当时也怀疑过是不是因为这个框架造成了性能的损失，从 `Instruments` 的测量数据来看，我们的顾虑是多余的， `ReactiveCocoa` 虽然使得整个函数调用栈的层次增加了不少，但是，这不是性能瓶颈。