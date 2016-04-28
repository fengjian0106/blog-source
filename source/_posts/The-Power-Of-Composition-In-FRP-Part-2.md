title: 用 FRP 事半功倍的写代码（二）
date: 2016-04-26 10:38:03
tags: [FRP, ReactiveCocoa] 
---

### 利用 map 组装顺序执行的业务
这其实应该是最常见的使用场景，有一类业务，是可以抽象成一组按顺序执行的串行任务的，比如下面这段代码

``` objectivec
/*
 ColdSignal<NSString, NSError>
 Completion: decode success;
 Error: FMBarCodeServiceErrorDomain || NSURLErrorDomain || RACSignalErrorDomain with RACSignalErrorTimedOut  //4
 Scheduler: specified;
 Multicast: NO;
 */
- (RACSignal *)decodeBarWithURLString: (NSString *)urlString {
    NSParameterAssert(urlString != nil);
    
    @weakify(self);
    return [[[self getUIImageWithURLString:urlString]  //1
             flattenMap:^(UIImage *image) {
                 @strongify(self);
                 return [self decodeBarWithUIImage:image];  //2
             }]
            timeout:1.5 onScheduler:[RACScheduler schedulerWithPriority:RACSchedulerPriorityDefault]];  //3
    
}
```

这段代码做的事情并不复杂，就是传入一个图片的 url 地址，然后下载对应的图片，然后尝试对这张图片进行二维码解码：

1. getUIImageWithURLString 里面完成的小任务，就是下载 UIImage。当下载失败的时候，会发出一个 NSURLErrorDomain 的 NSError。
2. 这里的小任务，就是对前一步得到的 UIImage 进行二维码解码。当解码失败的时候，会发出一个 FMBarCodeServiceErrorDomain 的 NSError(自己的业务代码中定义的 error domain)。
3. 这里的业务需求，是当用户长按一张图片的时候，弹出一个选项菜单，让用户可以选择合适的操作，比如『保存图片』，『转发图片』等等，同时，如果这张图片中能够识别出二维码，在弹出的选项菜单中，还要包含一项『识别图中二维码』。二维码解析是需要消耗一定的时间的，下载图片也是需要时间的，有些情况下，即便图片本身的确是一个二维码，但是二维码可能很复杂，解析的时间就会比较长，为了保证最佳的用户体验，这里需要做一个超时逻辑，如果 1.5 秒内都还没有解析出一个有效的二维码，则放弃当前的解析动作。[timeout](http://reactivex.io/documentation/operators/timeout.html) 操作就是针对这种场景的，当到达设定的超时时间时，如果还没有发送 Next 事件，则会在 Pipeline 中发送一个 RACSignalErrorDomain 的 NSError，error code 是 RACSignalErrorTimedOut。
4. 这个 Pipeline 是由好几个小任务组合出来的，每一个环节都有可能发送 error，所以对于这个 Pipeline 的订阅者，捕获到的 NSError 会是好几个不同 Domain 的其中之一。

这个 Pipeline 的订阅者的代码会是下面这种样子：

``` objectivec
-(void)jsCallImageClick:(NSString *)imageUrl imageClickName:(NSString *)imgClickName {
    NSMutableArray *components = [NSMutableArray arrayWithArray:[imageUrl componentsSeparatedByString:@"&qmSrc:"]];
    NSMutableArray *temp = [NSMutableArray arrayWithArray:[(NSString*)[components firstObject] componentsSeparatedByString:[NSString stringWithFormat:@"&%@:",imgClickName]]];
    [self filterJsArray:temp];
    NSString *imageUrlString = [NSString stringWithFormat:@"%@",(NSString *)[temp firstObject]];
    
    RACSignal *barCodeStringSignal = [self.barCodeService decodeBarWithURLString:imageUrlString];
    
    @weakify(self);
    [[barCodeStringSignal
      deliverOn:[RACScheduler mainThreadScheduler]]  //1
     subscribeNext:^(NSString *barCodeString) {
         @strongify(self);
         [self showImageSaveSheetWithImageUrl:imageUrl withImageClickName:imgClickName withBarCode:barCodeString];
     } error:^(NSError *error) {
         
         @strongify(self);
         [self showImageSaveSheetWithImageUrl:imageUrl withImageClickName:imgClickName withBarCode:nil];
     } completed:^{
     }];
}
```

因为 decodeBarWithURLString 的内部在使用 timeout 的时候，已经通过 RACScheduler 切换到了后台线程，所以在订阅者(UI)这里还要切换回 [RACScheduler mainThreadScheduler]。

### 捕获并且替换 error
下面也是一个真实业务场景中的代码片段，有适当的删减，需求大致可以描述为：FMContact.contactItems 数组里包含的是一个联系人的所有的 email 地址(至少有一个)，在用 FMContactCreateAvatarCell 显示这个联系人的头像的时候，要通过其中的一个 email 地址，构造出一个 url 地址，然后下载对应的头像，最后把头像 image 设置到 UIButton 上。

``` objectivec
//1
/*
 ColdSignal<UIImage?, NoError>
 Completion: download image finished;
 Error: 
 Scheduler: specified;
 Multicast: NO;
 */
- (RACSignal *)getAvatarWithContact: (FMContact *)contact {
    RACSignal *addrs = [[contact.contactItems.rac_sequence
                         map:^(FMContactItem *contactItem) {
                             return contactItem.email;
                         }]
                        signal];//4
    
    return [[[[addrs take:1]  //5
              map:^id(NSString *emailAddr) {
                  return [[[FMAvatarManager shareInstance] rac_asyncGetAvatar:emailAddr]
                          retry:3];  //6
              }]
             flatten]
            catch:^RACSignal *(NSError *error) {
                //7
                return [RACSignal return:nil]; //8
            }];
}

- (void)initPipelineWithCell:(FMContactCreateAvatarCell *)cell {
    @weakify(cell);
    [[[[self getAvatarWithContact:self.contact] //1
       deliverOnMainThread]
      takeUntil:cell.rac_prepareForReuseSignal]
     subscribeNext:^(UIImage *image) {
         @strongify(cell);
         if (image) { //2
             [cell.avatarButton setImage:image forState:UIControlStateNormal];
         }
     } error:^(NSError *error) {
         //3
     } completed:^{
     }];
}

```

这个业务需求看上去也没有太大的难度，大家肯定都可以用传统的代码写出来，但是如果用 FRP，则可以用声明式(declarative)的代码把逻辑写的更清晰：

1. getAvatarWithContact 定义了一个 Signal，通过输入参数 FMContact，获取一个对应的头像，如果头像下载成功，则通过 next 把 image 发送给 Pipeline 的订阅者，如果下载图片失败，并不会发送 error，而是在 next 里面发送一个 nil。
2. 这个 Pipeline 只会有一次 next 事件，按照 Signal 的定义，可能为 nil，所以需要检查。
3. 这个 Pipeline 是不会产生 error 的，所以这里不需要做任何事情。但是真正的下载图片的操作，也就是 [[FMAvatarManager shareInstance] rac_asyncGetAvatar:emailAddr] 这一句代码产生的 signal，是有 error 事件的，有意思的地方就是如何对这里可能出现的 error 进行处理，请接着往下看。
4. 把 FMContact.contactItems 数组里面的 email 地址，用 signal 的形式发送出来。
5. FMContact 至少有一个 email 地址，因为只需要显示一个头像，所以直接用最简单的办法，通过 [take](http://rxmarbles.com/#take) 操作取出其中的第一个 email 地址。
6. 从模块设计的角度来看，应该遵循一个基本原则，如果一个小任务可能出现失败的情况，就应该通过 error 把错误信息发送出去。[[FMAvatarManager shareInstance] rac_asyncGetAvatar:emailAddr] 是在下载头像图片，肯定是存在下载失败的可能性，所以这个小任务应该遵循这个基本原则。但是，为了更好的用户体验，可以在 Pipeline 中增加一个环节，添加一个策略，就是遇到下载失败的时候，自动重新下载一遍，总共尝试 3 次，这个需求可以用 [retry](http://reactivex.io/documentation/operators/retry.html) 操作方便的实现出来。
7. 如果运气真的不好，3 次下载都失败了，那 Pipeline 里还是会发送 error 的，但是 getAvatarWithContact 这个 signal 的设计要求是不要 error，这个时候就该用到 [catch](http://reactivex.io/documentation/operators/catch.html) 操作了。catch 做的事情，就是当 Pipeline 里出现 error 的时候，把这个 error 『吃掉』，然后用另外的一个 signal 来替换原来的 signal，让整个 Pipeline 可以继续发送 next 数据。
8. [RACSignal return:nil] 就是用来替换的 signal，这个 signal 会在 next 里面发送一次 nil，然后立刻就 complete。(如果业务需求变化，这里也可以通过 [RACSignal return:defaultAvatarImage] 发送一个默认的头像图片，Pipeline 是很方便的，可以灵活的组装)。
