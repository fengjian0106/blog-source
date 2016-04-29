title: 用 FRP 事半功倍的写代码（三）
date: 2016-04-28 11:31:36
tags: [FRP, ReactiveCocoa]
---

### collect + combineLatest 或者 zip

RAC 里面的 [collect](http://reactivex.io/documentation/operators/to.html) 是一个比较容易理解的操作，它的强大之处，在于和其他的操作进行组合之后，可以完成很复杂的业务逻辑。在看真实业务代码之前，先通过下面的代码初步了解一下这种 Pipeline 的行为模式。`collect 相当于 Rx 中的 ToArray 操作`

#### 版本 1

``` objectivec
- (void)testCollectSignalsAndCombineLatestOrZip {
    //1
    RACSignal *numbers = @[@(0), @(1), @(2)].rac_sequence.signal;
    
    RACSignal *letters1 = @[@"A", @"B", @"C"].rac_sequence.signal;
    RACSignal *letters2 = @[@"X", @"Y", @"Z"].rac_sequence.signal;
    RACSignal *letters3 = @[@"M", @"N"].rac_sequence.signal;
    NSArray *arrayOfSignal = @[letters1, letters2, letters3]; //2
    
    
    [[[numbers
       map:^id(NSNumber *n) {
           //3
           return arrayOfSignal[n.integerValue];
       }]
      collect]  //4
     subscribeNext:^(NSArray *array) {
         DDLogVerbose(@"%@, %@", [array class], array);
     } completed:^{
         DDLogVerbose(@"completed");
     }];
}
```

这个代码纯粹只是为了演示 collect 的行为模式：

1. 构造一个 NSNumber 的数组，包含数字 0、1、2，并且转换成 signal。
2. 用同样的方法，构造 3 个字符串的数组，并转换成 signal，再把得到的 3 个 signal 放到数组 arrayOfSignal 中。
3. 这里形成了一个 signal 的嵌套，但是和以前的处理方式不一样，并不会直接在后续环节中使用 flatten 操作，而是先使用 collect。
4. collect 操作会把 Pipeline 中所有的 next 发送的数据收集到一个 NSArray 中，然后一次性通过 next 发送给后续的环节。

这段代码的执行结果如下：

```
2016-04-28 17:45:38:034 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] __NSArrayM, (
    "<RACDynamicSignal: 0x7ffee1c9dc10> name: ",
    "<RACDynamicSignal: 0x7ffee1c9dda0> name: ",
    "<RACDynamicSignal: 0x7ffee1c9df20> name: "
)
2016-04-28 17:45:38:034 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] completed
```

可以看到，array 里面包含的是 3 个 signal。另外，因为 signal 已经形成嵌套了，所以迟早是要 flatten 的，那么如何 flatten 呢？

#### 版本 2

因为 array 里面有 3 个 signal，所以可以构造一种 Pipeline，把这 3 个 signal 合并成一个 signal，然后对合并后的 signal 再做 flatten 操作。合并的时候，可以有不同的策略，先看下面这段代码：

``` objectivec
- (void)testCollectSignalsAndCombineLatestOrZip {
    RACSignal *numbers = @[@(0), @(1), @(2)].rac_sequence.signal;
    
    RACSignal *letters1 = @[@"A", @"B", @"C"].rac_sequence.signal;
    RACSignal *letters2 = @[@"X", @"Y", @"Z"].rac_sequence.signal;
    RACSignal *letters3 = @[@"M", @"M"].rac_sequence.signal;
    NSArray *arrayOfSignal = @[letters1, letters2, letters3];
    
    
    [[[[numbers
        map:^id(NSNumber *n) {
            return arrayOfSignal[n.integerValue];
        }]
       collect]
      flattenMap:^RACStream *(NSArray *arrayOfSignal) {
          //1
          return [RACSignal combineLatest:arrayOfSignal
                                   reduce:^(NSString *first, NSString *second, NSString *third) {
                                       return [NSString stringWithFormat:@"%@-%@-%@", first, second, third];
                                   }];
      }]
     subscribeNext:^(NSString *x) {
         DDLogVerbose(@"%@, %@", [x class], x);
     } completed:^{
         DDLogVerbose(@"completed");
     }];
}
```

这段代码在接收到 collect 发送的 array 之后，对这个数组里面的 signal 进行了一个 [combineLatest](http://rxmarbles.com/#combineLatest) 操作，这个时候，原本的 3 个 signal 被 reduce 成了一个 signal，这个 signal 继续被 flatten 一次，然后最终被 Pipeline 的订阅者接收到。

这段代码的执行结果如下(也可能和下面的结果完全不一样，这是正常的，combineLatest 操作就是这样)：

```
2016-04-28 18:48:14:453 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] NSTaggedPointerString, A-Z-N
2016-04-28 18:48:14:453 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] NSTaggedPointerString, B-Z-N
2016-04-28 18:48:14:454 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] NSTaggedPointerString, C-Z-N
2016-04-28 18:48:14:455 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] completed
```

#### 版本 3

除了 combineLatest，[zip](http://rxmarbles.com/#zip) 操作也可以把多个 signal reduce 成一个，但是 zip 的策略是不一样的。

``` objectivec
- (void)testCollectSignalsAndCombineLatestOrZip {
    RACSignal *numbers = @[@(0), @(1), @(2)].rac_sequence.signal;
    
    RACSignal *letters1 = @[@"A", @"B", @"C"].rac_sequence.signal;
    RACSignal *letters2 = @[@"X", @"Y", @"Z"].rac_sequence.signal;
    RACSignal *letters3 = @[@"M", @"M"].rac_sequence.signal;
    NSArray *arrayOfSignal = @[letters1, letters2, letters3];
    
    
    [[[[numbers
        map:^id(NSNumber *n) {
            return arrayOfSignal[n.integerValue];//!! this is Signal, but just use map NOT flatMap
        }]
       collect]
      flattenMap:^RACStream *(NSArray *arrayOfSignal) {
          //1
          return [RACSignal zip:arrayOfSignal
                         reduce:^(NSString *first, NSString *second, NSString *third) {
                             return [NSString stringWithFormat:@"%@-%@-%@", first, second, third];
                             
                         }];
      }]
     subscribeNext:^(NSString *x) {
         DDLogVerbose(@"%@, %@", [x class], x);
     } completed:^{
         DDLogVerbose(@"completed");
     }];
}
```

这段代码的执行结果是下面这个样子，不像前面的 combineLatest，zip 操作的结果，只能出现下面这种唯一的情况：

```
2016-04-28 18:55:01:208 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] NSTaggedPointerString, A-X-M
2016-04-28 18:55:01:209 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] NSTaggedPointerString, B-Y-N
2016-04-28 18:55:01:209 [com.ReactiveCocoa.RACScheduler.backgroundScheduler] completed
```

### 保存联系人的头像

前面的代码很抽象，在业务中，能用上这种 Pipeline 吗？当然是可以的，比如下面这段代码：

``` objectivec
- (RACSignal *)savaAvatar:(UIImage *)image withContact:(FMContact *)contact {
    NSParameterAssert(image != nil);
    NSParameterAssert(contact.contactItems.count > 0);
    
    //1
    RACSignal *addrs = [[contact.contactItems.rac_sequence
                         map:^(FMContactItem *contactItem) {
                             return contactItem.email;
                         }]
                        signal];
    
    return [[[[addrs
               map:^id(NSString *emailAddr) {
                   return [[[[FMAvatarManager shareInstance] rac_setAvatar:emailAddr image:image] //2
                            map:^id(id value) {
                                //4
                                return RACTuplePack(value, nil);//rac_setAvatar成功的时候
                            }]
                           catch:^RACSignal *(NSError *error) {
                               //3
                               return [RACSignal return:RACTuplePack(nil, error)];
                           }];
               }]
              collect]  //5
             flattenMap:^RACStream *(NSArray<RACSignal *> *arrayOfSignal) {             
                 return [[RACSignal zip:arrayOfSignal]  //6
                         map:^id(RACTuple *tuple) {     //7
                             //8
                             return [tuple allObjects];
                         }];
             }]
            map:^id(NSArray<RACTuple *> *value) {
                //9
                return value;
            }];
}
```

这段代码稍微有点复杂，做的事情是让 FMContact 里面的所有 email 地址，和一个 image 关联在一起，并且保存在服务器端，关键是下面这几个点：

1. 把 contact.contactItems 里面所有的 email 转换成 signal 的形式发送出来。
2. 每次 map 的时候，得到一个 email 地址，调用 [[FMAvatarManager shareInstance] rac_setAvatar:emailAddr image:image] 让 email 地址和 image 关联在一起，这个接口也是返回一个 signal，当成功的时候，next 里面发送一个 value (业务中并不关心这个 value 的具体值，只关心是否成功)，如果失败，则会发送一个 error。如果不对 error 做特殊处理，当遇到一次 error 的时候，会使整个 Pipeline error，有些业务需要这种处理 error 的默认方式 (n 个小任务中，任何一个出现 error，整个 Pipeline 都要 error)，但是我们这里的业务，并不想要这种效果，如果一个 email 上的操作失败了，不希望整个 Pipeline 因为这个 error 而结束，而是要其余的 email 地址继续执行各自的小任务，等所有的 email 都处理完毕后，再由 Pipeline 的订阅者一起处理所有的 error，这个时候，就需要用到 catch 操作了。`这里有一个槽点，rac_setAvatar 每次都需要传入 image 和 email 地址，然后调用服务器接口进行保存操作，这种方式的接口，不够优雅，对于每一个 email 地址，都要重新发送一遍 image，也有点浪费流量，这是一个历史原因造成的问题。更好的方案是，先把 image 上传到服务器端，然后得到这个 image 对应的一个唯一值，比如 id，然后在这里，只需要让这个 image 的 id 和 email 能够关联起来就行了。不过这并不影响这里 Pipeline 的设计，不管是 image 还是 id，Pipeline 的形状是没有区别的。`
3. 在 catch 里面，用新的 signal 替换原有的 signal。因为需要把 error 暂存下来，放到最后再做处理，所以，用 RACTuple 把 error 包装起来并且发送出去。
4. 虽然目前的业务，并不关心 [[FMAvatarManager shareInstance] rac_setAvatar:emailAddr image:image] 发送的 next 数据，但是，把 next 发送的数据和 error 一起用 RACTuple 包装起来，也是一个合理的设计(万一以后需要用到这个值了呢)，当接收到 next 的时候，error 就是 nil，当发生 error 的时候，相当于 next 就是 nil，所以在这里，返回的是 RACTuplePack(value, nil)，而在前面 3 中，返回的是 RACTuplePack(nil, error)。
5. 使用 collect 操作。注意，前面 map 操作返回的是一个 signal，signal 的 next 发送的是一个 RACTuple，而 collect 发送的 next 是 NSArray&lt;RACSignal *&gt;。
6. 前面的 map 已经形成了 signal 的嵌套，而且还通过 collect 把嵌套的 signal 放到了数组里面，所以这里需要先把数组里的 signal 合并成一个，然后再 flatten 出来。zip 操作符合我们的需求。
7. 这里不像前面的代码演示那样使用 + (instancetype)zip:(id&lt;NSFastEnumeration&gt;)streams reduce:(id (^)())reduceBlock 接口，而是先使用 + (instancetype)zip:(id&lt;NSFastEnumeration&gt;)streams，然后 map，因为前一种 zip，输入参数 streams(数组) 中包含的元素的数量是已知的，所以可以直接在 reduce(变参数方法) 中把所有的参数都罗列出来，我们这里的 Pipeline，arrayOfSignal 里面的元素个数是不固定的，所以只能用原始的 zip 接口，然后在 map 中再进一步处理 zip 发送的 RACTuple。
8. 在这个 map 里面得到的 RACTuple 是 zip 操作返回的，这个 tuple 里面包含的每一个数据，是前面 4 里面返回的 RACTuple，这里的 RACTuple 里面又包含了 RACTuple，千万不要搞晕了。如果没搞清楚这里的数据到底是怎么来的，可以再倒回去看看前面的步骤。为了方便后续的处理，可以把外层 RACTuple 里面的数据放到一个 NSArray 里面，然后再返回给下一个环节。[tuple allObjects] 就是做的这个动作(其实 RACTuple 的内部，就是用 NSArray 存储的数据)。
9. 直接把 value 返回，让 Pipeline 的订阅者得到最终的结果。这里没有做任何额外的动作，仅仅是为了说明现在得到的数据是一个 NSArray&lt;RACTuple *&gt;。可以在这里加一些日志，方便调试。不执行这一次 map 操作也是可以的。

### 表单页面

再看另外一个真实业务，如下图：

![edit contact](/images/edit_contact.png)

这是一个编辑联系人的页面，整体是用 UITableView 实现的，可以动态的增加、删减字段，其中有一个需求，只有当至少有一个字段有数据的时候，右上角的『保存』按钮才可以使用。如果这个页面，不需要动态的增加、删减字段，那这个需求是很容易实现的，如果不使用 UITableView，就算要动态的增加、删减字段，这个需求实现起来也还好，不会很困难。但是现在的问题在于，要在 UITableView 的基础上实现，这就有点复杂了，UITableViewCell 是在复用的，所以不能直接依赖 UITableViewCell 里面的 UITextField 来判断『保存』按钮是否可用，必须严格的使用 MVC 的思路，先把 UI 上所有的操作(增加、删减字段，编辑字段内容)都映射到 model 上，通过 model 再来计算『保存』按钮是否可用。UITableView 的代码，是传统代码和 RAC 混合编写的，RAC 做的事情并不多，主要是把 UITextField 的内容用 signal 发送出来，因为并不复杂(但是也挺繁琐的，产品还提了很多很细节的体验要求)，所以这里不详细讨论，主要还是看一下基于 model 构造的 Pipeline：

``` objectivec
- (void)initPipline {
    @weakify(self);
    //1
    RACSignal *emailsIsNil = [[RACObserve(self.contact, contactItems) //2
                               flattenMap:^id(NSMutableArray *items) {
                                   if (items.count == 0) { //3
                                       return [RACSignal return:[NSNumber numberWithBool:YES]];
                                   }
                                   
                                   //4
                                   return [[[items.rac_sequence.signal
                                             map:^id(FMContactItem *item) {
                                                 return [[RACObserve(item, email)  //5
                                                          distinctUntilChanged]    //6
                                                         map:^id(NSString *email) {
                                                             //7
                                                             return [NSNumber numberWithBool:(email.length == 0)];
                                                         }];
                                             }]
                                            collect]  //8
                                           flattenMap:^id(NSArray *arrayOfBoolSignal) {
                                               return [[[RACSignal combineLatest:arrayOfBoolSignal]  //9
                                                        map:^id(RACTuple *tuple) {
                                                            //10
                                                            BOOL b = YES;
                                                            for (NSUInteger i = 0; i < tuple.count; i++) {
                                                                NSNumber *n = [tuple objectAtIndex:i];
                                                                b = b && n.boolValue;
                                                            }
                                                            return [NSNumber numberWithBool:b];
                                                        }]
                                                       distinctUntilChanged];  //11
                                           }];
                               }]
                              distinctUntilChanged];  //11
    
    //12
    RACSignal *phonesIsNil = [[RACObserve(self.contact, telephone)
                               map:^id(NSMutableArray *phones) {
                                   if (phones.count == 0) {
                                       return [NSNumber numberWithBool:YES];
                                   }
                                   
                                   for (NSString *phone in phones) {
                                       if (phone.length > 0) {
                                           return [NSNumber numberWithBool:NO];
                                       }
                                   }
                                   
                                   return [NSNumber numberWithBool:YES];
                               }]
                              distinctUntilChanged];
    
    RACSignal *addressIsNil = [[RACObserve(self.contact, familyAddress)
                                map:^id(NSMutableArray *addrs) {
                                    if (addrs.count == 0) {
                                        return [NSNumber numberWithBool:YES];
                                    }
                                    
                                    for (NSString *addr in addrs) {
                                        if (addr.length > 0) {
                                            return [NSNumber numberWithBool:NO];
                                        }
                                    }
                                    
                                    return [NSNumber numberWithBool:YES];
                                }]
                               distinctUntilChanged];
    
    RACSignal *customInfosIsNil = [[RACObserve(self.contact, customInformations)
                                    flattenMap:^id(NSMutableArray *infos) {
                                        if (infos.count == 0) {
                                            return [RACSignal return:[NSNumber numberWithBool:YES]];
                                        }
                                        
                                        return [[[infos.rac_sequence.signal
                                                  map:^id(FMCustomInformation *info) {
                                                      RACSignal *nameSignal = [[RACObserve(info, name)
                                                                                distinctUntilChanged]
                                                                               map:^id(NSString *name) {
                                                                                   return [NSNumber numberWithBool:(name.length == 0)];
                                                                               }];
                                                      
                                                      RACSignal *infoSignal = [[RACObserve(info, information)
                                                                                distinctUntilChanged]
                                                                               map:^id(NSString *i) {
                                                                                   return [NSNumber numberWithBool:(i.length == 0)];
                                                                               }];
                                                      
                                                      return [RACSignal combineLatest:@[nameSignal, infoSignal]
                                                                               reduce:(id)^id(NSNumber *name, NSNumber *info){
                                                                                   return [NSNumber numberWithBool:(name.boolValue && info.boolValue)];
                                                                               }];
                                                      
                                                  }]
                                                 collect]
                                                flattenMap:^id(NSArray *arrayOfBoolSignal) {
                                                    return [[[RACSignal combineLatest:arrayOfBoolSignal]
                                                             map:^id(RACTuple *tuple) {
                                                                 BOOL b = YES;
                                                                 for (NSUInteger i = 0; i < tuple.count; i++) {
                                                                     NSNumber *n = [tuple objectAtIndex:i];
                                                                     b = b && n.boolValue;
                                                                 }
                                                                 return [NSNumber numberWithBool:b];
                                                             }]
                                                            distinctUntilChanged];
                                                }];
                                    }]
                                   distinctUntilChanged];
    
    RACSignal *nickIsNil = [[RACObserve(self.contact, nick)
                             map:^id(NSString *nick) {
                                 @strongify(self);
                                 if (self.contact.nick == nil || [self.contact.nick isEqualToString:@""] == YES) {
                                     return [NSNumber numberWithBool:YES];
                                 }
                                 return [NSNumber numberWithBool:NO];
                             }]
                            distinctUntilChanged];
    
    RACSignal *markIsNil = [RACObserve(self.contact, mark)
                            map:^id(NSString *mark) {
                                @strongify(self);
                                if (self.contact.mark == nil || [self.contact.mark isEqualToString:@""] == YES) {
                                    return [NSNumber numberWithBool:YES];
                                }
                                return [NSNumber numberWithBool:NO];
                            }];
    
    RACSignal *birthdayIsNil = [RACObserve(self.contact, birthday)
                                map:^id(NSString *birthday) {
                                    @strongify(self);
                                    if (self.contact.birthday == nil || [self.contact.birthday isEqualToString:@""] == YES) {
                                        return [NSNumber numberWithBool:YES];
                                    }
                                    return [NSNumber numberWithBool:NO];
                                }];
    
    //13
    NSArray *allSignal = @[nickIsNil, emailsIsNil, markIsNil, phonesIsNil, addressIsNil, birthdayIsNil, customInfosIsNil];
    self.contactHasNoPros = [[[[RACSignal combineLatest:allSignal]  //13
                               map:^id(RACTuple *tuple) {
                                   //14
                                   BOOL b = YES;
                                   for (NSUInteger i = 0; i < tuple.count; i++) {
                                       NSNumber *n = [tuple objectAtIndex:i];
                                       b = b && n.boolValue;
                                   }
                                   return [NSNumber numberWithBool:b];
                               }]
                              distinctUntilChanged]
                             deliverOnMainThread];
}

```

这部分代码有点长，不过不用恐惧，中间有很大一部分代码都是做的类似事情，只需要看其中的一个就行，以 email 字段为例子：

![edit contact](/images/edit_email.png)

1. 联系人的字段，被划分为了好几个部分，比如 email 数组、电话号码数组、备注信息字段等等，每一部分的处理逻辑都是类似的，主要看一下 email 相关的部分。
2. 当添加或删除 email 的时候，UITableView 部分的代码，已经在 FMContact.contactItems 数组上做了对应的动作，这里通过 RACObserve 对这个 model 进行 KVO，就可以获取到 FMContactItem 的数组。
3. 如果用户删除了所有的 email 地址(FMContactItem 数组的元素个数为 0)，emailsIsNil 就应该为 YES，说明当前输入的 email 是没有值的。
4. 如果 FMContactItem 数组的元素个数不为 0，则把这个数组里面的 FMContactItem 转换成 signal 的形式发送出去。
5. UI 模块会实时的更新 FMContactItem.email 字段，所以这里也是使用 RACObserve 监听 email 字段的值。
6. [distinctUntilChanged](http://rxmarbles.com/#distinctUntilChanged) 操作相当于一种过滤，只有当这一次 next 发送的数据和前一次 next 发送的数据不一样的时候，才会把这次 next 发送的数据继续往后续环节传递。
7. 拿到一个 email 地址的时候，只要这个 email 的长度大于 0，就认为这个字段是有值的(并没有进行 email 有效性检查，即便输入的 email 不合理，『保存』按钮仍然可用，只有点击『保存』按钮的时候，才会检查 email 是否合理有效，产品需求是设计成这样的)。
8. 使用 collect。注意前面 5 所在的 map 操作，返回的是 signal，所以这里形成了 signal 的嵌套，然后 collect 又会把这些 signal 全部放到一个数组里面。
9. 拿到 signal 的数组后，要把这些 signal 合并成一个，combineLatest 满足这里的需求。
10. 这里实现具体的产品需求，比如现在有 n 个 email 的输入框，当所有的输入框都没有输入内容的时候，才认为 email 是没有值的，只要有任何一个 email 输入框有内容，都认为 email 是有值的。
11. 这几个地方使用 distinctUntilChanged，都是为了避免不必要的 signal 数据传递。
12. 这里好几个 signal，都是类似的思考思路和实现方式。
13. 把不同的 *IsNil signal 放到一个数组里，用 combineLatest 把它们合并成一个。
14. 和 10 类似，实现产品约定好的需求，当所有输入框都没有内容的时候，这个联系人就是没有值的(通过 self.contactHasNoPros 这个 signal 来传递这个 Bool 值)。

上面这段代码，最终实现出了一个 signal，就是 contactHasNoPros，这个 signal 的订阅者，根据 next 发送的 Bool 值，设置 button 的状态就可以了，代码片段如下：

``` objectivec
@weakify(self);
[[self.contactEditView.contactHasNoPros
      not] //1
     subscribeNext:^(NSNumber *x) {
         @strongify(self);
         self.navigationItem.rightBarButtonItem.enabled = x.boolValue;
     }];
```

因为 contactHasNoPros 发送 YES 的时候，表达的含义是联系人所有的字段都没有值，没有值的时候，『保存』按钮应该是不可用状态，所以这里用 not 操作先做一个 Bool 值的取反，然后再设置 button 的 enabled 状态。
