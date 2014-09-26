title: 微信web页面开发代码规范以及最佳实践
date: 2014-09-26 13:41:32
tags: [html, css, sass, flexbox, mvc, mvvm]
---

本来专职是做iOS开发的，对 `node.js` 感兴趣，所以也学习过服务器开发的一些皮毛。前段时间公司的一个产品，要覆盖iOS、Android和微信web这3个前端，微信web页面的第一个版本，是服务器端同事用最传统的web技术做出来的，也就是使用后端模板渲染的技术组装页面，前端js则使用了 `jQuery` 做简单的操作。东西做出来后，体验特别的不好，尤其是每次页面跳转都要加载一个新页面，会有延迟。后来狠下决心重做一遍，完全采用前端渲染的技术。当时本来就缺人手，好歹我也算是会用JavaScript，义不容辞的自然也就把这个项目接手过来了。

做的过程很累，特别的赶进度，而且是摸着石头过河，未知的因素很多。还好iOS做的很熟悉了，很多经验或问题，可以直接照搬到web端，借助Google和Stack Overflow，记录了一大堆笔记出来。前两天另一个团队启动新项目，微信web端也是要做的，同事便找到我，想要一些经验分享，琢磨了一下，笔记写的比较凌乱，毕竟主要是给自己看的，符合自己的思维模式习惯，但是并不适合给别人看，干脆整理一份出来，方便别人查看，对自己也是再次清理一下思路。

### CSS中如何命名class
这个话题没有唯一答案，而且特别的松散，风格很多。

目前团队里使用 <http://webuild.envato.com/blog/how-to-scale-and-maintain-legacy-css-with-sass-and-smacss/> 这种方案，但是并不完全遵照它的代码模板，比如我们不使用 `Sass` 语法，而是用 `SCSS` 语法。

另外还推荐阅读 <http://blog.jobbole.com/47702/> 这个里面提到了几乎所有的技术、框架、思路，比如 `Sass`、`BEM`、`OOCSS`、`SMACSS`、`OrganicCSS`等等。

### 使用 `Sass` 编写CSS
这个没啥多说的，语法并不难。引入了编程语言中的一些思路和技巧，对于iOS和Android开发者来说，很容易上手。

### CSS书写规范 && 顺序

参照 <http://markdotto.com/2011/11/29/css-property-order/> 以及 <http://codegeekz.com/standardizing-css-property-order/>

### 如何计算 `CSS Box Model` 中的 `width` 和 `height`
我们使用 `CSS3` 中的 [`border-box`](http://www.w3school.com.cn/cssref/pr_box-sizing.asp) 模型，这个就和iOS中的 `UIView` 的尺寸模型保持了一致，也更直观，容易理解。 

### 使用 `Flexbox` 做 Layout
项目的前期，为了实现一些手机上常见的布局，我们大量使用了CSS中的 `float` 、`table-cell` 等等，但是代码会比较复杂，而且可读性不好。直到发现了神器 [`Flexbox`](http://css-tricks.com/snippets/css/a-guide-to-flexbox/)。

另外，还有一个基于 `Sass` 的工具 <https://github.com/mastastealth/sass-flex-mixin.git>，帮助我们更好的进行编码。

### 使用 `Yeoman` 实现工作流
不懂得使用工具的web开发者，不是好前端，嘿嘿 ^_^


### `MV*` 框架 Or `jQuery` 类型的库
* 单纯从技术角度来看，[AngularJS](https://angularjs.org/) 是最合适的框架，而且针对mobile，还有基于 `AngularJS` 的 [Ionic](http://ionicframework.com/) 框架。`Ionic` 是对手机适配的最好的框架(没有之一)。但是 `Ionic` 的体积比较大，官方宣传时定义其为 *framework for developing hybrid mobile apps*。如果是对网速不敏感的使用场景，或者网速很快的场景，其实 `Ionic` 是可以做 `web app` 的。
* 百度开源的 [gmu](http://gmu.baidu.com/)，是类似于 `jQuery UI` 的库，但是是基于 `zepto` 的，很轻量级，而且也提供了不少的 `widget`。但是为了轻量，并没有套用 `MV*` 模式，所以应用场景复杂的时候，代码通常会组织的比较凌乱。交互界面复杂的时候，还会暴露出各种各样的坑，比如click事件穿透，就让我们大吃苦头。
* 我个人已经不太愿意继续使用 `gmu` 了，如果**_真正只需要开发轻量级的页面_**，我宁愿直接用 `Sass` + `zepto` 或 <http://minifiedjs.com/> 来实现。
* [Backbone](http://backbonejs.org/) 是相对轻量级的 `MVC` 框架，在体积大小和功能上有合理的舍取，但是框架本身只注重设计模式的引入，并不包含一套完整的针对mobile的 `widget`，所以我们还整理了另外一种思路，就是基于 `Backbone`，再加上各种各样小的lib，根据需求组合起来使用。这种方案可能存在的问题就是这些lib各自为政，不像 `AngularJS` 这种框架一样都在一个体系内协同工作，所以开发的时候也许会有很多坑，得做一遍才会有深刻的体会。有两个例子，非常值得学习参考，<https://github.com/ccoenraets/directory-backbone-ratchet> 和 <http://n12v.com/2-way-data-binding/>。

### 常见`widget`
按照iOS平台上的开发经验，针对mobile，常见的 `widget` 包括这些(有官方SDK自带的，也有大量第三方开源的，iOS平台现在很完善，有很多 `widget` 可以拿来即用)

* button，textinput，slider，progress bar，image view，switch等等，这些是最常见的 `widget`
* 全屏HUD，比如iOS上的 <https://github.com/jdg/MBProgressHUD>
* 全屏的菜单选择类 `widget`，比如iOS自带的 `UIAlertView` 和 `UIActionSheet`
* popview，比如iOS上的 <https://github.com/chrismiles/CMPopTipView.git>
* 免干扰式的下拉信息提示框，比如iOS上的 <https://github.com/toursprung/TSMessages.git>
* 系统级的页面切换方式，比如iOS自带的 `UINavigationController` 和 `UITabBarController`，以及第三方开源的 <https://github.com/ECSlidingViewController/ECSlidingViewController.git>

mobile web平台上，`widget` 的生态环境并不好。相对而言，[Ionic](http://ionicframework.com/) 自带的`widget`是最完善的，而且有框架的支持，也更容易实现自定义的 `widget`。唯一的问题就是 `Ionic` 框架比较大。

`Backbone` 或 `gmu` 中，除了最常见的 `widget` 外，其他的通常都只能自己实现，比如在使用 `gmu` 的时候，我们就只能自己编写[`ActionSheet`](https://github.com/fengjian0106/actionsheet.git)、全屏HUD、免干扰式的下拉信息提示框、以及系统级的页面导航控制器。由于缺少框架级的支持，除了 [`ActionSheet`](https://github.com/fengjian0106/actionsheet.git) 外，其他几个 `widget` 的代码实现都很粗暴，而且遇到了各种各样的bug。

实际项目中，最好从一开始做交互设计的时候，就考虑 `widget` 的问题，尽量使用最常见的 `widget`，舍弃一些复杂的交互方式。