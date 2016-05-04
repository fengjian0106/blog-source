title: 用 ReactiveCocoa 事半功倍的写代码（一）
date: 2016-04-17 21:25:59
tags: [FRP, ReactiveCocoa]
---

### 前言
[FRP](https://en.wikipedia.org/wiki/Functional_reactive_programming) 是一门学习曲线比较陡峭的技术，回想自己以前的学习过程，也是反反复复好几次，而且总是挫败感很强。不过还好坚持了下来，现在也算是用着比较顺手了。

关于 FRP， 最容易被吐槽的地方就是没有好的学习资料和文档。一开始我也是这种感觉，后来在反复尝试的过程中，发现其实真的不是文档的问题。先说我的结论 ---- 不要指望脱离代码能够把 FRP 的原理讲清楚，这是 FRP 和其他编程技术的一个明显差异，这就类似于很难用一段文字把一个数学公式描述清楚一样。而且，即便是开始看用 FRP 编写的各种代码了，还是会觉得太抽象了，仍然需要大量的时间体会代码，或者说，『悟』出其中的一些基本门道。


关于入门学习，没有捷径，最好的办法就是通过代码来学习，下面是我觉得比较好的一些入门学习资料

* [The introduction to Reactive Programming you've been missing](https://gist.github.com/staltz/868e7e9bc2a7b8c1f754)
* [ReactiveCocoa Documentation](https://github.com/ReactiveCocoa/ReactiveCocoa/tree/master/Documentation) *我本人主要是做 iOS 开发，目前使用的是 RAC 这个库，所以它的官方文档也是一个学习途径。另外，本文中的代码也是使用 RAC 进行编写*
* [ReactiveCocoa Tutorial – The Definitive Introduction: Part 1/2](https://www.raywenderlich.com/62699/reactivecocoa-tutorial-pt1)
* [ReactiveCocoa Tutorial – The Definitive Introduction: Part 2/2](https://www.raywenderlich.com/62796/reactivecocoa-tutorial-pt2)
* [Interactive diagrams of Rx Observables](http://rxmarbles.com/) *这个是一组动态效果图，用可视化的效果演示了一些 FRP 里常用操作(当然，其实还是很抽象的)*

之所以说 FRP 的学习曲线很陡峭，不仅仅是指它的入门学习比较耗时费脑，当入了门或者稍微找到一些感觉之后，紧接着就会面对第二个问题：FRP 里面提供的都是一些比较抽象的函数操作，怎样才能用这些基本函数来解决各种各样的业务问题？尤其是那些很抽象的操作，怎样才能用起来？

这个系列的文章，主要就是针对后面这第二个问题，做的一些 demo 演示。

可以把 FRP 看成是一种更高级的 [Pipeline](http://www.dossier-andreas.net/software_architecture/pipe_and_filter.html) 编程范式，Pipeline 的一个精髓，就是可以灵活的组合，虽然 FRP 里常用的操作也就那么几十个，但是一旦像搭积木那样对它们进行了组装之后，FRP 的强大之处一下子就展现了出来。

FRP 通常是以库或框架的形式提供给使用者，目前已经有很多常见编程语言的具体实现。在这个系列文章中，将使用 RAC 2 (ReactiveCocoa 的 Objective-C 版本) 进行编写。但是 FRP 本质上是一种编程范式，从 Pipeline 的角度来看，它的侧重点在于如何组装出不同形状的 Pipeline，而不太在乎 Pipeline 的具体构成材料(编程语言)，从框架的角度来看，虽然有不同语言版本的实现，但是每个版本里，提供的诸如 map、flattenMap、reduce 等基础操作，在概念上和行为模式上，又都是一样的。所以，FRP 也是一门 "Learn once, write anywhere" 的技术。

FRP 有几个明显的好处，比如可以减少中间状态变量的使用，可以编写紧凑的代码，可以用同步风格编写异步运行的代码，在本系列文章中，也会尽量体现出这些特点。


### 处理键盘的弹出和隐藏
这个业务其实是非常简单的，就是在某个 UIViewController 里面，当检测到键盘弹出的时候，为了避免键盘遮挡住某个 UIView，需要根据键盘的高度重新对 view 进行 layout，用 RAC 写出来的代码是下面这个样子：

``` objectivec
//1
- (void)initPipeline {
    @weakify(self);
    RACSignal *keyboardWillShowNotification =
    [[NSNotificationCenter.defaultCenter rac_addObserverForName:UIKeyboardWillShowNotification object:nil]
     map:^id(NSNotification *notification) {
         //2
         NSDictionary* userInfo = [notification userInfo];
         NSValue* aValue = [userInfo objectForKey:UIKeyboardFrameEndUserInfoKey];
         return aValue;
     }];
    
    [[[[[NSNotificationCenter.defaultCenter rac_addObserverForName:UIKeyboardWillHideNotification object:nil]
        map:^id(NSNotification *notification) {
            //3
            return [NSValue valueWithCGRect:CGRectZero];
        }]
       merge:keyboardWillShowNotification]   //4
      takeUntil:self.rac_willDeallocSignal]  //6
     subscribeNext:^(NSValue *value) {
         NSLog(@"Keyboard size is: %@", value);
         //5
         @strongify(self);
         self.messageEditViewContainerViewBottomConstraint.constant = 5.0 + [value CGRectValue].size.height;
         
         [self.view updateConstraints];
         [UIView animateWithDuration:0.6 animations:^{
             @strongify(self);
             [self.view layoutIfNeeded];
         }];
     } completed:^{
         //6
         NSLog(@"%s, Keyboard Notification signal completed", __PRETTY_FUNCTION__);
     }];
}
```

用数字标注的地方，是比较关键的点：

1. 很多时候，Pipeline 都是只需要构建一次的，如果是针对 UIViewController，通常都是在 viewDidLoad 方法里调用 [self initPipeline]，如果是针对 UIView，则很有可能是在 awakeFromNib 方法里进行调用，这里遵循的一个策略是，在模块『活』起来之后，应该尽快的构造所有的 Pipeline，如果是 model 或 service 类型的模块，则很可能是在 init 完成后，就调用 initPipeline，但是对于 UI 性质的模块，因为有 iOS 平台相关的 view 加载策略，而且 Pipeline 通常又是和 UI 交互有关，所以通常是需要在 view 生命期相关的方法中才构造 Pipeline。
2. 通过 map 操作，把 UIKeyboardWillShowNotification 转换成一个 CGRect(包装在 NSValue 里面)。[map](http://rxmarbles.com/#map) 操作是 FRP 里面最核心的一个基本操作，也是最体现函数式编程(FP)哲学的一个操作，所谓的这个哲学，用通俗的话来描述，就是『把复杂的业务拆分成一个一个的**小任务**，每一个小任务，都需要一个输入值，并且会给出一个输出值(当然也会反馈错误信息)，而且每个小任务都只专心的做一件事情』。如果第一个小任务的输出值，是第二个小任务的输入值，那么，就可以用 map 操作把这两个小任务串联在一起。在接收到 UIKeyboardWillShowNotification 消息通知的时候，这个小任务的输入值就是 NSNotification，输出值是键盘尺寸对应的 CGRect，小任务本身做的事情，就是从 NSNotification 里面取出包装着这个 CGRect 的 NSValue。
3. 当接收到 UIKeyboardWillHideNotification 消息通知的时候，这个小任务要做的事情，和 2 里面的小任务是类似的，只不过这一次，NSNotification 并没有包含键盘的尺寸，那我们自己用 CGRectZero 构造一个就行了。
4. 终于到了这段代码的重点了，[merge](http://rxmarbles.com/#merge) 操作在这里的使用效果，相当于把 2 和 3 里面的两个小任务的输出值作为自己的输入值，按照时间先后顺序排列起来，然后作为自己这个小任务的输出值，返回给 Pipeline 中的下一个环节。`这样描述还是很抽象，看不懂，是吧？没关系，早就说过用语言很难描述了。把代码运行起来，通过 NSLog(@"Keyboard size is: %@", value) 这句代码的输出信息体会一下 merge 的实际效果。`
5. 这里才是真正的实现业务想要的效果，根据前一个小任务的输出值(键盘尺寸 CGRect)来计算 layout 的尺寸。
6. [takeUntil](http://rxmarbles.com/#takeUntil) 是一个难点，如果没有这一句代码调用，运行代码后会发现，前面 5 里面的业务还是正常执行了，但是当 self 被 dealloc 后(比如 pop UIViewController 后)，NSLog(@"Keyboard size is: %@", value) 这句代码还是会被执行到(因为已经处理过 retain cycle，所以此时 self 是 nil)，这是因为当 self 被 dealloc 后，这个 Pipeline 并没有被释放，Pipeline 里面还是有数据在继续流动。这个话题牵扯到 RAC 框架中的[内存管理策略](https://github.com/ReactiveCocoa/ReactiveCocoa/blob/master/Documentation/Legacy/MemoryManagement.md)，很重要，后面的内容中还会讲到这个话题。这里暂时只需要知道可以借助 takeUntil:self.rac_willDeallocSignal 这样的一行代码方便的解决问题就行了。

### Singal上的 next、complete、error
在学习的过程中，发现有一个问题很容易被忽略掉，那就是 Signal 的 next、complete、error 这 3 种数据，会在什么时候被发送出来，针对这个问题做过一个总结，放在了 [这篇文档](https://github.com/ziipin-code/ZiipinTemplateProject/blob/master/rac.md) 中，主要目的是使用一种简单易懂的格式把 Signal 的关键信息描述出来，这里简单摘录一下。

#### 基础格式
```
HotSignal<T, E>   // or ColdSignal<T, E>
Completion: ...
Error: ...
Scheduler: ...
Multicast: ...
```

#### 关键字解释
*   [HotSignal And ColdSignal](https://github.com/ReactiveCocoa/ReactiveCocoa/blob/master/Documentation/Legacy/DesignGuidelines.md#use-descriptive-declarations-for-methods-and-properties-that-return-a-signal):
    *   `HotSignal`: Signal 已经处于活动状态(activated);
    *   `ColdSignal`: Signal 需要订阅(subscribed)才会活动(activate);
*   T: Signal sendNext 的类型, 可以下面几种情况:
    *   `T`: 表示只会发送 1 次 next 事件, 内容是类型 `T` 的实例;
    *   `T?`:  表示只会发送 1 次 next 事件, 内容是类型 `T` 的实例或者 `nil`;
    *   `[T]`: 表示会发送 0 到 n 次 next 事件, 内容是类型 `T` 的实例;
    *   `[T?]`: 表示会发送 0 到 n 次 next 事件, 内容是类型 `T` 的实例或者 `nil`;
    *   `None`: 表示**不会**发送 next 事件;
*   E: Signal sendError 的类型, 通常是 `NSError` 或 `NoError`; `NoError` 表示 Signal 不会 sendError;
*   Completion: 描述什么情况 sendCompleted;
    *   如果 next 事件的发送次数是 `无穷多次`，相当于使用者永远也接收不到 Completed 事件，所以这一行可以不写;
*   Error:  描述什么情况 sendError;
    如果 Signal 不会 sendError, 这一行可以不写;
*   Scheduler: Signal 所在的线程，通常是 `main` `specified` `current`, 默认是 `current`
    * main 模块内部的pipeline有切换不同的scheduler，所以模块内部有责任确保最终的signal始终是在main schedular上的
    * specified 模块内部自定义了一个任务队列，模块会确保最终返回的signal都在这个特定的schedular中(或者是使用全局默认的后台schedular)
    * current 模块内部pipeline没有做任何scheduler的切换，且不指定特定的schedular，所以最终返回的signal和外部调用者的线程保持一致
*   Multicast: 是否广播，通常是 `YES` `NO`, 默认是 `NO`

#### 所有可能出现的有意义的非嵌套 Signal  
```
*    HotSignal<T, NoError>
*    HotSignal<T?, NoError>
*    HotSignal<[T], NoError>
*    HotSignal<[T?], NoError>
*    HotSignal<None, NoError>
*    HotSignal<T, NSError>
*    HotSignal<T?, NSError>
*    HotSignal<[T], NSError>
*    HotSignal<[T?], NSError>
*    HotSignal<None, NSError>

*    ColdSignal<T, NoError>
*    ColdSignal<T?, NoError>
*    ColdSignal<[T], NoError>
*    ColdSignal<[T?], NoError>
*    ColdSignal<None, NoError>
*    ColdSignal<T, NSError>
*    ColdSignal<T?, NSError>
*    ColdSignal<[T], NSError>
*    ColdSignal<[T?], NSError>
*    ColdSignal<None, NSError>
```


### 发送验证码的倒计时按钮
![Retry Button](/images/retryButton.png)

如上图，这里的需求是，点击右上角的按钮后，该按钮不可以使用，同时在按钮上显示一个倒计时时间，当达到倒计时时间后，按钮恢复可用状态。这个需求并不难，相信大家都可以写出来，但是，每个人写出来的代码，风格肯定千差万别，而且，免不了会需要一些状态变量来记录一些信息，比如定时器对象和倒计时的时间等等。如果换用 RAC，则可以在一段连续的代码中，满足所有的需求，代码如下：

``` objectivec
//1
/*
 ColdSignal<RACTuple<NSString, NSNumber<BOOL> >), NoError>
 Completion: 1分钟倒计时结束;
 Error: none;
 Scheduler: main;
 Multicast: NO;
 */
- (RACSignal *)retryButtonTitleAndEnable {
    static const NSInteger n = 60;
    
    RACSignal *timer = [[[RACSignal interval:1 onScheduler:[RACScheduler mainThreadScheduler]]  //7
                         map:^id(id value) {
                             return nil; //8
                         }]
                        startWith:nil]; //9
    
    //10
    NSMutableArray *numbers = [[NSMutableArray alloc] init];
    for (NSInteger i = n; i >= 0; i--) {
        [numbers addObject:[NSNumber numberWithInteger:i]];
    }
    
    return [[[[[numbers.rac_sequence.signal zipWith:timer]  //11
               map:^id(RACTuple *tuple) {
                   //12
                   NSNumber *number = tuple.first;
                   NSInteger count = number.integerValue;
                   
                   if (count == 0) {
                       return RACTuplePack(@"重试", [NSNumber numberWithBool:YES]);
                   } else {
                       NSString *title = [NSString stringWithFormat:@"重试(%lds)", (long)count];
                       return RACTuplePack(title, [NSNumber numberWithBool:NO]);
                   }
               }]
              takeUntil:[self rac_willDeallocSignal]] //13
             setNameWithFormat:@"%s, retryButtonTitleAndEnable signal", __PRETTY_FUNCTION__]
            logCompleted]; //14
}

- (void)initPipeline {
    @weakify(self);
    [[[[[[self.retryButtton rac_signalForControlEvents:UIControlEventTouchUpInside]
         map:^id(id value) {
             //2
             @strongify(self);
             return [self retryButtonTitleAndEnable];
         }]
        startWith:[self retryButtonTitleAndEnable]]  //3
       switchToLatest]  //4
      takeUntil:[self rac_willDeallocSignal]]  //5
     subscribeNext:^(RACTuple *tuple) {
         //6
         @strongify(self);
         NSString *title = tuple.first;
         [self.retryButtton setTitle:title forState:UIControlStateNormal];
         self.retryButtton.enabled = ((NSNumber *)tuple.second).boolValue;
     } completed:^{
         //5
         NSLog(@"%s, pipeline completed", __PRETTY_FUNCTION__);
     }];
    
    //这里省略了点击 retryButtton 后具体要做的业务逻辑，同时也省略了验证按钮和验证码输入框的处理逻辑
}
```

对关键代码的描述如下：

1. 设计一个 RACSignal，这个 Signal 每次发送的 Next 数据里面包含的就是按钮上要显示的文本信息和按钮的可用状态。从模块的角度来看，这个 Signal 的内部细节(倒计时逻辑)，外部使用者是不需要知道的，所以后面我们会先看外层 Pipeline 的实现代码，然后再倒回来看这个 Signal 的内部逻辑。
2. 每当 retryButtton 被点击的时候，要重新启动一个定时器，所以在这个 map 操作里面，调用 [self retryButtonTitleAndEnable] 得到一个 Signal，将这个 Signal 作为这个小任务的输出值。注意，因为这里 map 操作返回的是一个 Signal，形成了一个 Pipeline 的嵌套，所以可以预见到，在外层 Pipeline 的后续操作中，肯定是需要把这个内嵌的 Pipeline flatten 出来的。
3. 在业务需求中，点击这个 retryButtton 后，要请求服务器发送一个验证码(省略了这部分的代码，如果要用 RAC 实现的话，是比较容易的)，同时，当每次进入这个 UI 页面的时候，不需要用户主动点击这个 retryButtton 按钮，首先就要自动的请求服务器发送一个验证码，这种情况下，也要求 retryButtton 开始进入倒计时的模式，所以，用 [startWith](http://rxmarbles.com/#startWith) 操作，在外层 Pipeline 中先插入第一个 Next 数据，因为是同样的倒计时逻辑，所以这里也是调用 [self retryButtonTitleAndEnable] 得到内嵌的 Pipeline。
4. 前面已经提过了，既然形成了 Pipeline 的嵌套，那肯定是要把这种嵌套解出来的，这里使用 [switchToLatest](http://reactivex.io/documentation/operators/images/switch.c.png) 更合适。要注意区分一下和 [flattenMap](http://reactivex.io/documentation/operators/flatmap.html) 的差异。
5. Pipeline 的生命期控制，前面的例子中已经讲过这种技巧了，但是，这是写上这句，只是一个双保险。复杂的地方在于外层 Pipeline 有 switchToLatest 操作，这个 switchToLatest 后的 Signal 什么时候才会 Completed，请继续看至后面 13 中的解释。
6. 这里是更新 retryButtton 的 title 和状态。
7. 现在开始回到内层 Pipeline 的逻辑中去。用 Pipeline 的方式实现一个定时器，借助 RAC 提供的 interval 操作就行。每隔一秒都会在主线程上发送一个 Next。
8. 7 里面的定时器上的 Next 数据，是当前的系统时间值，我们的需求里面并不需要这个时间值，所以这里直接 map 成 nil。
9. RACSignal interval 要隔一秒后才会发出第一次，需要用 startWith 立刻发送一个，代表倒计时的初始值。
10. 把倒计时要用到的数字放到一个数组里面，然后通过 numbers.rac_sequence.signal 语句转换成一个 Signal。
11. 把前面 10 中得到的 Signal 和 9 中得到的 timer Signal，用 [zipWith](http://rxmarbles.com/#zip) 组装起来。注意一点，这个通过 zipWith 组装出来的 Signal，会在 numbers.rac_sequence.signal Completed 的时候 Completed (这句话有点绕，需要结合 zipWith 的定义仔细体会一下)。
12. 根据倒计时的数值，计算按钮上需要显示的 title 信息和按钮的状态。
13. 前面 11 里面的 zipWith 操作，可以确保倒计时结束时，会触发 Completed，但是万一在倒计时的过程中，用户离开了当前页面，这个时候就需要通过 takeUntil 来触发 Completed。之所以在这里这么注重 Completed，是因为前面的 5 里面的 switchToLatest 操作，会 `sends completed when both the receiver and the last sent signal complete`。
14. 通过 setNameWithFormat 和 logCompleted 打印一些 log 信息，方便调试，注意观察一下 Signal 的 Completed。

### 内存管理，自动释放 Pipeline
从前面的 code 中可以看到，好几个地方都在强调要触发 Completed，这完全就是为了正确的进行内存管理，避免内存泄露，[避免手动的调用 disposal](https://github.com/ReactiveCocoa/ReactiveCocoa/blob/7877f99bdfb4be1c82c4804082e99c35d0a93a91/Documentation/Legacy/DesignGuidelines.md#avoid-explicit-subscriptions-and-disposal)。takeUntil:self.rac_willDeallocSignal 是一种常用的手段。

还有一种典型的场景，也可以通过 takeUntil 操作来触发 Completed，代码如下：

``` objectivec
- (UICollectionViewCell *)collectionView:(UICollectionView *)collectionView cellForItemAtIndexPath:(NSIndexPath *)indexPath {
    WWKPhoto *photo = self.photos[indexPath.row];
    XMCollectionImageViewCell *cell = [self.imageCollectionView dequeueReusableCellWithReuseIdentifier:NSStringFromClass([XMCollectionImageViewCell class]) forIndexPath:indexPath];
    cell.imageView.image = photo.thumbnail;
    
    
    @weakify(self);
    [[[cell.longPressSignal map:^id(XMCollectionImageViewCell *viewCell) {
        @strongify(self);
        return [self.imageCollectionView indexPathForCell:viewCell];
    }]
      takeUntil:[cell rac_prepareForReuseSignal]]
     subscribeNext:^(NSIndexPath *longPressIndexPath) {
         @strongify(self);
         UIAlertController *alert= [UIAlertController
                                    alertControllerWithTitle:@"确定删除此图片"
                                    message:nil
                                    preferredStyle:UIAlertControllerStyleAlert];
         
         UIAlertAction* ok = [UIAlertAction actionWithTitle:@"确定" style:UIAlertActionStyleDefault
                                                    handler:^(UIAlertAction * action){
                                                        @strongify(self);
                                                        [[self mutableArrayValueForKey:@keypath(self, photos)] removeObjectAtIndex:longPressIndexPath.row];
                                                        [self.imageCollectionView deleteItemsAtIndexPaths:@[longPressIndexPath]];
                                                    }];
         UIAlertAction* cancel = [UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleDefault
                                                        handler:^(UIAlertAction * action) {
                                                            [alert dismissViewControllerAnimated:YES completion:nil];
                                                        }];
         
         [alert addAction:ok];
         [alert addAction:cancel];
         
         [self.containerViewController presentViewController:alert animated:YES completion:nil];
     } completed:^{
     }];
    
    return cell;
}
```

这段代码也很简单，唯一需要特别注意的就是 takeUntil:[cell rac_prepareForReuseSignal] 这一句，因为 UICollectionViewCell 本身是有一套复用机制的，每个 cell 上的 Pipeline 的生命期和 cell 本身的生命期并不一致，所以不能依赖于 cell.rac_willDeallocSignal，而应该使用 [cell rac_prepareForReuseSignal] 这个更准确的 Signal。


讨论到这里，还可以得到一个结论，在设计 Signal 的时候，要尽量的让这个 Signal 能够发送 Completed 事件，这样才能够充分的利用 Pipeline 的自动释放功能，保持代码的简洁。RAC 框架里，有一些很常用的 Signal，其实它们的内部实现也是用类似 takeUntil 的操作做了这种处理，比如下面这些 Signal：
``` objectivec
@interface UIControl (RACSignalSupport)
- (RACSignal *)rac_signalForControlEvents:(UIControlEvents)controlEvents;
@end

@interface UIGestureRecognizer (RACSignalSupport)
- (RACSignal *)rac_gestureSignal;
@end

RACObserve 宏定义
```

下面这个 Signal，则是没有 Completed 事件的，要求它的使用者来决定什么时候释放对应的 Pipeline：
``` objectivec
@interface NSNotificationCenter (RACSupport)
- (RACSignal *)rac_addObserverForName:(NSString *)notificationName object:(id)object;
@end
```









