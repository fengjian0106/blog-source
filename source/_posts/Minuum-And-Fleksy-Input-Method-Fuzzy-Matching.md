title: 如何实现Minuum和Fleksy输入法中的智能纠错功能
date: 2015-02-11 10:56:30
tags: [iOS, keyborad, kNN]
---

输入法的产品一直在持续技术演进，最近的一项工作，是实现了一个类似 [Minuum](http://minuum.com/) 和 [Fleksy](http://fleksy.com/) 这两款输入法中的模糊输入功能的单词智能纠错引擎，前后尝试过4种不同的算法思路，最终才找到适合手机的解决方案，特此记录一下。

1. 基于贝叶斯推断的，主要线索是 [http://norvig.com/spell-correct.html](http://norvig.com/spell-correct.html)
2. 基于Levenshtein自动机的，主要线索是 [http://blog.notdot.net/2010/07/Damn-Cool-Algorithms-Levenshtein-Automata](http://blog.notdot.net/2010/07/Damn-Cool-Algorithms-Levenshtein-Automata)
3. 一种基于预处理词库的改进算法，主要线索是 [http://blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/](http://blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/)
4. 使用机器学习中的kNN算法，主要线索是 [http://minuum.com/model-your-users-algorithms-behind-the-minuum-keyboard/](http://minuum.com/model-your-users-algorithms-behind-the-minuum-keyboard/) 和 [http://www.zhihu.com/question/27567987](http://www.zhihu.com/question/27567987)

前三个方案，都是用传统的算法思路，基于[编辑距离](http://en.wikipedia.org/wiki/Levenshtein_distance)来实现模糊匹配，但是在手机上无法满足输入法的性能需求，尤其是查询速度这一点，而且也无法做到和Minuum或Fleksy类似的纠错效果。最终的第4个方案，则是彻底更换了思路，直接用机器学习中的 [kNN](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) 算法，把字符串映射到更抽象的几何空间中，也就是所谓的特征向量，进行纯粹的数学计算。学习和研究的过程中，是直接用Python做的代码原型验证，放到github上了，有兴趣的朋友可以看看 [https://github.com/fengjian0106/Minuum-Fleksy-Fuzzy-Matching](https://github.com/fengjian0106/Minuum-Fleksy-Fuzzy-Matching)
