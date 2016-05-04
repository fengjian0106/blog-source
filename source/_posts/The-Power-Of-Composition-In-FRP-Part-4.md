title: 用 ReactiveCocoa 事半功倍的写代码（四）
date: 2016-05-03 14:42:56
tags: [FRP, ReactiveCocoa]
---

### 监听系统截屏操作的复杂管道

这是一个很复杂的 Pipeline，因为要做的业务比较繁琐，如下图：

![monitor screenshot](/images/monitor_screenshot.png)

需求大致可以描述为：

1. 当 app 停留在读信页面的时候，要实时的监听用户是否有截屏操作。
2. 在 1 的基础上，只有 app 前台运行的时候，才实时监听用户是否有截屏操作，如果是后台状态，则不监听。
3. 如果用户有截图动作，则将截图内容显示在一个预览视图内(如上图中红框区域)。
4. 如果用户点击了预览视图，则进入后续的业务流程，对截图进行涂鸦编辑等等。
5. 如果点击了预览视图的外部区域，则隐藏预览视图。
6. 如果 10 秒钟之内没有任何操作，也自动隐藏预览视图。

主要的代码如下：

``` objectivec
//13
- (void)viewDidAppear:(BOOL)animated {
    [self initPipeline];
}

- (void)initPipeline {
    //1
    RACSignal *isNotActive = [[NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationWillResignActiveNotification object:nil]
                              map:^id(NSNotification *notification) {
                                  return [NSNumber numberWithBool:NO];
                              }];
    
    RACSignal *isActive = [[[[NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationDidBecomeActiveNotification object:nil]
                             map:^id(NSNotification *notification) {
                                 return [NSNumber numberWithBool:YES];
                             }]
                            startWith:[NSNumber numberWithBool:YES]]
                           merge:isNotActive];
    
    RACSignal *isNotInBackground = [[NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil]
                                    map:^id(NSNotification *notification) {
                                        return [NSNumber numberWithBool:NO];
                                    }];
    
    RACSignal *isInForeground = [[[[NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationWillEnterForegroundNotification object:nil]
                                   map:^id(NSNotification *notification) {
                                       return [NSNumber numberWithBool:YES];
                                   }]
                                  startWith:[NSNumber numberWithBool:YES]]
                                 merge:isNotInBackground];
    
    //2
    RACSignal *didTakeScreenshot = [NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationUserDidTakeScreenshotNotification object:nil];
    
    @weakify(self);
    RACSignal *imageSignal = [[[[[[[RACSignal if:[RACSignal merge:@[isInForeground, isActive]] then:didTakeScreenshot else:[RACSignal never]] //3
                                   takeUntil:self.rac_willDeallocSignal]
                                  filter:^BOOL(id value) {
                                      //4
                                      @strongify(self);
                                      return [self filterScreenshotNotification];
                                  }]
                                 filter:^BOOL(id value) {
                                     //5
                                     @strongify(self);
                                     return self.previewShotView == nil;
                                 }]
                                map:^id(NSNotification *notification) {
                                    //6
                                    @strongify(self);
                                    return [self takeCurrentScreenshotOfWebview];
                                }]
                               multicast:[RACReplaySubject subject]]
                              autoconnect];//7
    
    //8
    RACSignal *hotSignalForPreview = [[[imageSignal
                                        map:^id(UIImage *image) {
                                            @strongify(self);
                                            return [self showScreenshotPreviewView:image];
                                        }]
                                       multicast:[RACReplaySubject subject]]
                                      autoconnect];
    
    //9
    RACSignal *cancel = [[hotSignalForPreview
                          map:^id(FMScreenshotPreviewView *previewView) {
                              return [previewView.cancelSignal
                                      map:^id(id value) {
                                          return nil;
                                      }];
                          }]
                         switchToLatest];
    
    //10
    RACSignal *editImage = [[hotSignalForPreview
                             map:^id(FMScreenshotPreviewView *previewView) {
                                 return [previewView.editImage
                                         map:^id(id value) {
                                             return nil;
                                         }];
                             }]
                            switchToLatest];
    
    //11
    RACSignal *otherActionForHidePreview = [[hotSignalForPreview
                                             map:^id(id value) {
                                                 RACSignal *willResignActive = [[[NSNotificationCenter.defaultCenter rac_addObserverForName:UIApplicationWillResignActiveNotification object:nil]
                                                                                 take:1]
                                                                                takeUntil:[RACSignal merge:@[cancel, editImage]]];
                                                 
                                                 RACSignal *timeout = [[[RACSignal return:nil]
                                                                        delay:10.0]
                                                                       takeUntil:[RACSignal merge:@[cancel, editImage, willResignActive]]];
                                                 
                                                 
                                                 return [[RACSignal merge:@[timeout, willResignActive]]
                                                         take:1];
                                             }]
                                            switchToLatest];
    
    //12
    RACSignal *shouldHidePreviewView = [RACSignal merge:@[cancel, editImage, otherActionForHidePreview]];
    
    //13
    RACSignal *viewWillDisappear = [self rac_signalForSelector:@selector(viewWillDisappear:)];
    
    //14
    [[[shouldHidePreviewView
       zipWith:hotSignalForPreview]
      takeUntil:viewWillDisappear]//13
     subscribeNext:^(RACTuple *tuple) {
         @strongify(self);
         [self hideScreenshotPreviewView:tuple];
     } completed:^{
     }];
    
    //15
    [[[imageSignal sample:editImage]
      takeUntil:viewWillDisappear]
     subscribeNext:^(UIImage *image) {
         @strongify(self);
         [self showDrawViewController:image];
     } completed:^{
     }];
}
```

代码有点长，而且里面的 signal 也比较多，主要是下面这些点：

1. 把 app 的 avtive、background、foreground 状态用 signal 的形式表达出来，使用 merge 操作把互为相反状态的 signal 合并在了一起，注意，还使用了 startWith 操作提供初始值。
2. 这个 signal 是真正的截屏操作，它仅仅是整个 Pipeline 中的一个小环节。
3. 因为只有 app 前台运行的时候才需要监听截屏事件，所以这里用 if/else 操作做第一层过滤。
4. 这里是第二层过滤，因为这个 ViewController 里面有很多功能，可能会出现一些页面层叠的情况，比如显示了一个 UIActionSheet 或自定义的菜单选项等等，这个时候，也是不需要监听截屏事件的。
5. self.previewShotView 就是显示预览图的 view，当已经有一个预览图正在显示的时候，也不需要监听截屏事件。
6. 终于过滤完了，按照产品的需求，并不是从系统相册里把用户刚才的截图找出来，而是在 app 中自行截图一遍(只截取有效区域，不截取导航栏和工具栏区域)，takeCurrentScreenshotOfWebview 方法返回的就是截图得到的 UIImage。
7. Pipeline 的后续部分，不止一处会用到前面得到的 UIImage，所以需要 hot signal。
8. 显示预览 view，同时在 Pipeline 中传递这个 view，这个也是 hot signal。
9. 点击预览 view 外部区域的时候，会发送 cancelSignal signal，因为形成了 signal 的嵌套，所以要通过 switchToLatest 取出来。
10. 类似的，点击预览 view 的时候，会发送 editImage signal，也是通过 switchToLatest 取出来。
11. 当已经显示了一个预览 view 的时候，如果超过10秒没有任何操作，或者 app 进入了不活跃状态，也是需要隐藏预览 view 的，这里组装出对应的 signal。注意这里是如何通过 takeUntil 控制 willResignActive 和 timeout 的生命期的。
12. 用 merge 操作组装出最终用来隐藏预览 view 的 signal。
13. 把这个 ViewController 的 viewWillDisappear 转换成 signal 的形式。根据需求，只有这个 ViewController 可见的时候，才监听截图事件，所以，在 viewDidAppear 的时候构造 Pipeline，在 viewWillDisappear 的时候释放 Pipeline。
14. 这里是隐藏预览 view 的具体逻辑。
15. 当点击了预览 view 的时候，通过 showDrawViewController 方法进入后续的业务逻辑，这里使用了 [sample](http://rxmarbles.com/#sample) 操作。
