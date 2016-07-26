title: 用 ReactiveCocoa 事半功倍的写代码（五）
date: 2016-07-25 16:40:42
tags: [FRP, ReactiveCocoa]
---

### 体会 Composition 的含义

有些读者可能会注意到一点，这个系列教程的英文标题是 The Power Of Composition In FRP，看上去并不像是中文标题的直接翻译，其实这也是纠结过后的一个妥协的选择，其实我个人更喜欢这个英文标题，因为 Composition 这个词，更能体现出 FRP 的一个精髓理念，如果要用一个中文词语来表示，我觉得『组装』这个词更准确一些。

先看下面这段代码：

```
- (void)fetchNecessaryDataForAccounts:(NSArray<FMAccount *> *)accounts {
    NSParameterAssert(accounts != nil);
    NSParameterAssert(accounts.count > 0);
    
    @weakify(self);
    [[[[[accounts.rac_sequence signal] //1
        map:^id(FMAccount *account) {
            //2
            return [[[QHOldAccountMigration fetchInitialDataForAccount:account]
                     map:^id(FMAccount *account) {
                         //3
                         return RACTuplePack(account, nil);
                     }]
                    catch:^RACSignal *(NSError *error) {
                        //3
                        return [RACSignal return:RACTuplePack(account, error)];
                    }];
        }]
       collect] //4
      flattenMap:^RACStream *(NSArray *arrayOfSignal) {
          return [[RACSignal zip:arrayOfSignal] //4
                  map:^id(RACTuple *tuple) {
                      NSMutableArray *successAccounts = [[NSMutableArray alloc] init];
                      NSMutableArray *failAccounts = [[NSMutableArray alloc] init];
                      
                      for (int i = 0; i < tuple.count; i++) {
                          RACTuple *t = [tuple objectAtIndex:i];
                          FMAccount *account = t.first;
                          NSError *error = t.second;
                          
                          if (error) {
                              [failAccounts addObject:account];
                          } else {
                              [successAccounts addObject:account];
                          }
                      }
                      
                      return RACTuplePack([successAccounts copy], [failAccounts copy]);
                  }];
      }]
     subscribeNext:^(RACTuple *tuple) {
         @strongify(self);
         NSArray *successAccounts = tuple.first;
         NSArray *failAccounts = tuple.second;
         //5
         
         if (failAccounts.count == 0) {
             [self jumpToOriginalLogic];
         } else {
             NSMutableString *title;
             if (successAccounts.count == 0) {
                 title = [[NSMutableString alloc] initWithString:@"所有账号迁移失败，请重新登录"];
             } else {
                 title = [[NSMutableString alloc] initWithString:@"邮箱账号"];
                 for (FMAccount *account in failAccounts) {
                     [title appendFormat:@"%@, ", account.profile.mailAddress];
                 }
                 
                 title = [[title substringToIndex:title.length - 2] mutableCopy];
                 [title appendString:@"迁移失败，需要重新登录"];
             }
             
             UIAlertController *alertController = [UIAlertController alertControllerWithTitle:title message:nil preferredStyle:UIAlertControllerStyleAlert];
             [alertController addAction:[UIAlertAction actionWithTitle:@"确定" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
                 @strongify(self);
                 for (FMAccount *account in failAccounts) {
                     FMMigrationFailAccount *failAccount = [FMMigrationFailAccount convertAccountToMigrationFailAccount:account];
                     [failAccount save];
                     
                     [[FMManager shareInstance] deleteAccount:account.accountId];
                 }
                 [self jumpToOriginalLogic];
             }]];
             
             UIViewController *viewController = [UIApplication sharedApplication].keyWindow.rootViewController;
             [viewController presentViewController:alertController animated:YES completion:^{
             }];
         }
     } error:^(NSError *error) {
         
     } completed:^{
         
     }];
}
```

这个 pipeline 其实用的就是 [collect + combineLatest 或者 zip](http://fengjian0106.github.io/2016/04/28/The-Power-Of-Composition-In-FRP-Part-3/) 这种管道模型，只不过管道内部具体的业务不一样，这里的业务就是针对每一个 FMAccount 帐号，下载一些必要的初始数据，然后等每个下载都完成后，再执行后续的业务，主要就是下面几个点：

1. 把包含有 FMAccount 的数组，先变换成 signal。
2. [QHOldAccountMigration fetchInitialDataForAccount:account] 里面执行的是下载初始数据的具体业务逻辑，其实它的内部就是一个用多个 map 操作串联起来的 pipeline，对应下载初始数据过程中的多个步骤。
3. 如果 fetchInitialDataForAccount 失败，则把 error 转换成 next 事件，用 tuple 的形式继续向 pipeline 的后续环节传递，等所有的下载都结束后，会统一对 error 进行处理。
4. 套用 collect + zip 这种 pipeline 模型。
5. 当所有的下载都结束后，才会运行到这里，successAccounts 和 failAccounts 分别对应成功下载初始数据的所有帐号和下载数据失败了的所有帐号，至于后面 if 分支里的代码，只是后续的一些业务逻辑功能，读者可以不用在意，我们的重点还是在于这个 pipeline 的外形。

在编写程序的时候，通常我们都会提到『复用』这个概念，最简单的场景就是函数复用，这里的 pipeline 也是一种复用，只不过 pipeline 不像普通函数那样通过抽象出输入参数和返回结果来实现复用，pipeline 的复用体现在管道的形状上，这里所谓的形状，就是把 FRP 中对 signal 的各种操作组装起来后 pipeline 的形状。多个 map 串联是一种形状，collect + zip 是一种形状，之前的教程中提到的那些案例，都可以理解为一种形状(甚至还可以看成是多个不同形状的 pipeline 的进一步组装)，每一种形状的管道，有输入的数据，有输出的数据，同时，还存在各种各样的中间处理环节，每次复用 pipeline 的时候，输入数据、输出数据以及中间处理环节，都是可以根据具体的业务需求灵活的进行填充的。

回到前面这个例子，accounts 是 pipeline 的输入，successAccounts 和 failAccounts 是 pipeline 的输出，其他的操作都可以看成是中间处理环节。这个 pipeline 仅仅是完成了下载数据的功能，在真实的产品需求中，为了更好的照顾用户体验，还希望能够显示出下载进度信息，也就是说，对于 accounts 这个输入，还需要另外一种形式的输出信息，可以体现出下载进度情况。这里还有一个约束条件，[QHOldAccountMigration fetchInitialDataForAccount:account] 这个操作本身是无法表现出下载数据时的进度信息的，因为并不是下载一个文件(在编程惯例中，通常只在上传和下载文件的时候或类似的场景中，才会设计出能体现进度信息的 API)，所以这里还需要想办法模拟出一种进度信息用来在 UI 上进行显示，主要代码如下：

```
- (void)fetchNecessaryDataForAccounts:(NSArray<FMAccount *> *)accounts {
    NSParameterAssert(accounts != nil);
    NSParameterAssert(accounts.count > 0);
    
    @weakify(self);
    //1
    RACSignal *fetchAllInitialData = [[[[accounts.rac_sequence signal]
                                        map:^id(FMAccount *account) {
                                            return [[[[[QHOldAccountMigration fetchInitialDataForAccount:account]
                                                       map:^id(FMAccount *account) {
                                                           return RACTuplePack(account, nil);
                                                       }]
                                                      catch:^RACSignal *(NSError *error) {
                                                          return [RACSignal return:RACTuplePack(account, error)];
                                                      }]
                                                     multicast:[RACReplaySubject subject]] //3
                                                    autoconnect];
                                        }]
                                       multicast:[RACReplaySubject subject]] //2
                                      autoconnect];
    
    
    //4
    RACSignal *businessLogicSignal = [[fetchAllInitialData collect]
                                      flattenMap:^RACStream *(NSArray *arrayOfSignal) {
                                          return [[RACSignal zip:arrayOfSignal]
                                                  map:^id(RACTuple *tuple) {
                                                      NSMutableArray *successAccounts = [[NSMutableArray alloc] init];
                                                      NSMutableArray *failAccounts = [[NSMutableArray alloc] init];
                                                      
                                                      for (int i = 0; i < tuple.count; i++) {
                                                          RACTuple *t = [tuple objectAtIndex:i];
                                                          FMAccount *account = t.first;
                                                          NSError *error = t.second;
                                                          
                                                          if (error) {
                                                              [failAccounts addObject:account];
                                                          } else {
                                                              [successAccounts addObject:account];
                                                          }
                                                      }
                                                      
                                                      return RACTuplePack([successAccounts copy], [failAccounts copy]);
                                                  }];
                                      }];
    
    
    [businessLogicSignal
     subscribeNext:^(RACTuple *tuple) {
         @strongify(self);
         NSArray *successAccounts = tuple.first;
         NSArray *failAccounts = tuple.second;
         //5
         if (failAccounts.count == 0) {
             [self jumpToOriginalLogic];
         } else {
             NSMutableString *title;
             if (successAccounts.count == 0) {
                 title = [[NSMutableString alloc] initWithString:@"所有账号迁移失败，请重新登录"];
             } else {
                 title = [[NSMutableString alloc] initWithString:@"邮箱账号"];
                 for (FMAccount *account in failAccounts) {
                     [title appendFormat:@"%@, ", account.profile.mailAddress];
                 }
                 
                 title = [[title substringToIndex:title.length - 2] mutableCopy];
                 [title appendString:@"迁移失败，需要重新登录"];
             }
             
             UIAlertController *alertController = [UIAlertController alertControllerWithTitle:title message:nil preferredStyle:UIAlertControllerStyleAlert];
             [alertController addAction:[UIAlertAction actionWithTitle:@"确定" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
                 @strongify(self);
                 for (FMAccount *account in failAccounts) {
                     FMMigrationFailAccount *failAccount = [FMMigrationFailAccount convertAccountToMigrationFailAccount:account];
                     [failAccount save];
                     
                     [[FMManager shareInstance] deleteAccount:account.accountId];
                 }
                 [self jumpToOriginalLogic];
             }]];
             
             UIViewController *viewController = [UIApplication sharedApplication].keyWindow.rootViewController;
             [viewController presentViewController:alertController animated:YES completion:^{
             }];
         }
     } error:^(NSError *error) {
         
     } completed:^{
         
     }];
    
    
    //11
    static const CGFloat tickCount = 60 / 0.5;
    RACSignal *timer = [[RACSignal interval:0.5 onScheduler:[RACScheduler mainThreadScheduler]]
                        map:^id(id value) {
                            return nil;
                        }];
    
    NSMutableArray *numbers = [[NSMutableArray alloc] init];
    for (NSInteger i = 0; i < tickCount; i++) {
        [numbers addObject:@(i)];
    }
    
    RACSignal *counter = [[[[[numbers.rac_sequence signal]
                             zipWith:timer]
                            map:^id(RACTuple *tuple) {
                                NSNumber *n = tuple.first;
                                return RACTuplePack(n, @(tickCount), nil);//12
                            }]
                           takeUntil:businessLogicSignal]
                          logCompleted];
    
    
    
    //6
    NSMutableArray *sequence = [[NSMutableArray alloc] init];
    for (int i = 0; i < accounts.count; i++) {
        [sequence addObject:@(i + 1)];
    }
    
    
    static NSInteger progressValue = 0;
    
    [[[[[[[fetchAllInitialData
           flatten]//6
          map:^id(RACTuple *tuple) {
              //7
              return tuple.first;
          }]
         zipWith:[sequence.rac_sequence signal]]//8
        combineLatestWith:[RACSignal return:@(accounts.count)]]//8
       map:^id(RACTuple *tuple) {
           //9
           RACTuple *nestedTuple = tuple.first;
           NSNumber *accountsCount = tuple.second;
           
           FMAccount *account = nestedTuple.first;
           NSNumber *order = nestedTuple.second;
           
           //10
           return RACTuplePack(order, accountsCount, account);
       }]
      merge:counter]//11
     subscribeNext:^(RACTuple *tuple) {
         NSNumber *order = tuple.first;
         NSNumber *accountsCount = tuple.second;
         FMAccount *account = tuple.third;
         
         //13
         if (account) {
             NSLog(@"fetch initial data finished, order is: [%@, %@], account is: %@", order, accountsCount, account.profile.mailAddress);
             
             NSInteger nextValue = order.integerValue * 100 / accountsCount.integerValue;
             if (order.integerValue == accountsCount.integerValue) {
                 nextValue = 100;
                 progressValue = 100;
             }
             
             if (nextValue > progressValue) {
                 progressValue = nextValue;
             }
         } else {
             NSLog(@"counter info, [%@, %@]", order, accountsCount);
             progressValue = progressValue + 1.0;
             
             //14
             if (progressValue > 95) {
                 progressValue = 95.0;
             }
         }
         
         //14
         NSLog(@"======== progressValue is: %ld", (long)progressValue);
         
     } error:^(NSError *error) {
     } completed:^{
     }];
}
```

下面看看这个 pipeline 是如何组装出来的：

1. 需求越复杂，通常 pipeline 也就会越复杂，现在我们遇到了新的需求，但是之前那个 pipeline 做的业务仍然是需要保留的，这种时候，通常可以考虑先把 pipeline 的代码拆分一下，然后对拆出来的 signal 或者 pipeline 重新进行组装。首先就可以把 [QHOldAccountMigration fetchInitialDataForAccount:account] 动作拆分出来，注意一点，这里还没有调用 collect 操作。
2. 因为后续多个业务逻辑都要用到前面第一步得到的 signal，根据业务的需求，对于每个 FMAccount 只需要下载一次数据，所以这里应该让 signal 变成广播的形式。
3. 内层嵌套的 signal 才是真正的 fetchInitialDataForAccount 动作，同理，也需要变成广播(其实在刚开始设计 pipeline 的时候，可能还意识不到需要广播，这种时候，可以先组装业务流程，当遇到问题后，再考虑是否需要使用广播 signal)。如果暂时看不明白为什么 2 和 3 两处需要使用广播，没有关系，先接着往后看，把整个 pipeline 看明白后，再倒回来想想为什么需要广播。
4. 这个中间环节也拆分出来，以备后用。
5. 这里是对 successAccounts 和 failAccounts 的处理逻辑，和前一个版本的 pipeline 没有区别。
6. 现在开始考虑如何显示进度信息，虽然每次 [QHOldAccountMigration fetchInitialDataForAccount:account] 调用是没有进度信息的，但是当有多次调用的时候，是可以计算出一种形式的进度信息的，比如总共有 5 个 FMAccount，当第一个 FMAccount 下载完数据(或者失败)的时候，整体进度就是 1/5，当第二个 FMAccount 下载完数据(或者失败)的时候，整体进度就是 2/5，依次类推。
7. 回忆一下 fetchAllInitialData 里面的内容，因为现在是计算进度信息，并不关心具体的 error，所以这里的 map 操作只需要返回 tuple.first，也就是只需要继续传递 FMAccount。
8. 这里连续调用 zip 和 combineLatest，如果觉得这里很难理解，没有关系，先分别回忆一下 zip 和 combineLatest 的效果，想象一下这里应该得到什么样的结果。
9. 前面的 zip 操作会得到一个 tuple，然后这个 tuple 又和 [RACSignal return:@(accounts.count)] 进行一次 combineLatest，所以这里会得到一个嵌套的 tuple。
10. 8 和 9 的操作，最终就是为了组装出这样的一个 tuple，然后继续在 pipeline 中传递。比如总共有 5 个 FMAccount，当第一个 FMAccount 下载完数据(或者失败)的时候，这个 tuple 的值是 (1, 5, 第一个 FMAccount 对象的指针)，当第二个 FMAccount 下载完数据(或者失败)的时候，返回的 tuple 的值是 (2, 5, 第二个 FMAccount 对象的指针)，依次类推，后续还会返回 3 个 tuple。
11. 前面已经组装出进度信息了，但是对于 UI 来说，这种进度信息还是太粗糙了，为了让 UI 上的进度条能够更平滑的进行动画过渡，还应该插入一些更细粒度的进度信息。这里借助 RAC 的定时器来构造出一种和 10 里面的 tuple 具有相同格式的 tuple 数据。关于这部分定时器的 pipeline，和 [发送验证码的倒计时按钮](http://fengjian0106.github.io/2016/04/17/The-Power-Of-Composition-In-FRP-Part-1/#发送验证码的倒计时按钮) 里面的 pipeline 是相似的形状的，可以看看之前的介绍。
12. 为了和 10 里面返回的 tuple 具有同样的格式，这里需要这样组装数据，按照顺序，这里返回的 tuple 依次将会是 (1, 120, nil)、(2, 120, nil)、(3, 120, nil)，依次类推，直到 (120, 120, nil)。
13. 终于到了 pipeline 的最终输出了，把 tuple 里面的数据先分别取出来，如果 account 不为 nil，则是通过 fetchAllInitialData 计算出来的进度信息，如果 account 为 nil，则对应通过定时器模拟出来的进度信息。假设最终的进度值会达到 100，这里还需要采用适当的手段将两种不同的进度值融合在一起，现在就是用最简单的办法进行的处理。
14. 如果定时器返回的 tuple 已经达到 (120, 120, nil)，而 fetchAllInitialData 还没有执行结束，这种情况下，不应该让进度值达到 100，必须得等所有的 fetchAllInitialData 都结束后进度值才能是 100，所以这里做一个约束，定时器模拟出的进度值，最大只能达到 95。

