title: CSP Concurrency Patterns In Swift
date: 2016-04-10 22:45:16
tags: [Swift, CSP, Concurrency]
---

### 前言
这篇文章的主要内容，是从 [Go Concurrency Patterns](https://talks.golang.org/2012/concurrency.slide) 翻译过来的。

原文是介绍 [Golang](http://golang.info/) 里面的 [CSP](https://en.wikipedia.org/wiki/Communicating_sequential_processes) 并发模型(Communicating Sequential Processes)，这里则是使用一个基于 Swift3.0 的库 [Venice](https://github.com/VeniceX/Venice) 编写的代码。

这篇文章的主要目的，并不是鼓励大家立刻就用 Swift 进行后端开发(至少不是目前这个阶段)，但是，对于想尝试全栈开发的 iOS 工程师来说，则可以通过这篇文章入门学习 CSP 这种并发编程模型。

*2016/04/08 update: 为了成功运行本文中的代码，需要安装 [https://swift.org/builds/development/xcode/swift-DEVELOPMENT-SNAPSHOT-2016-03-24-a/swift-DEVELOPMENT-SNAPSHOT-2016-03-24-a-osx.pkg](https://swift.org/builds/development/xcode/swift-DEVELOPMENT-SNAPSHOT-2016-03-24-a/swift-DEVELOPMENT-SNAPSHOT-2016-03-24-a-osx.pkg) 这个版本的 swift。*

*2016/04/13 update: [Venice](https://github.com/VeniceX/Venice) 里面的 Channel 不再支持基于自定义运算符的读写操作，只能使用 func api。*

### 为什么要讨论并发 (concurrency)
观察一下我们周围，能发现什么？

我们的世界里发生的事情，总是一步一步按顺序执行的吗？

或者说，发生在我们身边的所有的事件，是一个很复杂的组合体，里面充满了更独立、更小型的事件单元，这些单元之间，则是有各种各样的交互和组织关系。

其实就像后者描述的这样，顺序处理 (Sequential processing) 并不是完美的建模思路。

### 什么是并发？
并发是独立的计算任务的组合。

并发是一种软件的设计模式，用并发的思维模式，可以编写出更清晰的代码。

### 并发 (concurrency) 不是并行 (parallelism)
并发不是并行，但是可以在并行的基础上形成并发。

如果只有一个单核处理器(单线程模式)，则谈不上并行，但是仍然可以写出并发的代码。

另一方面，如果一段代码已经按照并发的思路进行了设计，那它也是可以很容易的在多核处理器(多线程模式)中并行执行。

*关于这个话题，更详细的讨论可以参看 [Concurrency is not Parallelism](http://talks.golang.org/2012/waza.slide)*

### 什么是好的代码架构
* 要容易理解
* 要容易使用
* 要容易描述出设计意图
* 不需要人人都是专家 (不应该总是出现大量threads，semaphores，locks，barriers等等高深的话题)

### CSP 的历史
CSP 并不是新技术，Communicating Sequential Processes 是 Tony Hoare 在 1978 年就提出来的概念，甚至在更早的 1975 年，Edsger Dijkstra 的 Guarded Command Language 里面，也能看到 CSP 的影子。

还有其他的一些语言，也有类似的并发模型


* Occam (May, 1983)
* Erlang (Armstrong, 1986)
* Newsqueak (Pike, 1988)
* Concurrent ML (Reppy, 1993)
* Alef (Winterbottom, 1995)
* Limbo (Dorward, Pike, Winterbottom, 1996).

### Venice / Golang 和 Erlang 的差异
Venice / Golang 通过 channels 来实现 CSP。

Erlang 是最接近于原始的 CSP 定义的，通过 name 进行通信，而非 channel。

它们的模型其实是一致的，只不过具体的表现形式有差异。

粗略来看相当于：writing to a file by name (process, Erlang) vs. writing to a file descriptor (channel, Venice / Golang).

### CSP 的基本使用
这篇文章最主要的目的是讨论并发模式，为了避免陷入编程语言本身的各种细节，我们只会使用到 Swift 很少的语法特性。

##### 从下面这个简单的 boring 函数开始

```
import Foundation

private func boring(msg: String) {
    for i in 0...10 {
        print("\(msg) \(i)")
        usleep(100)
    }
}

public func run01() {
    boring("this is a boring func")
}
```

很容易想象到，这段代码的执行结果会是下买这个样子

```
 this is a boring func 0
 this is a boring func 1
 this is a boring func 2
 this is a boring func 3
 this is a boring func 4
 this is a boring func 5
 this is a boring func 6
 this is a boring func 7
 this is a boring func 8
 this is a boring func 9
 this is a boring func 10
```

##### 稍微改动一下
增加一点随机的延时，让 message 出现的时机不可预测 (延迟时间仍然控制在1秒之内)。并且让 boring 函数一直循环运行。

```
import Foundation

private func boring(msg: String) {
    for i in 0..<Int.max {
        print("\(msg) \(i)")
        usleep(1000 * (arc4random_uniform(1000) + 1))
    }
}

public func run02() {
    boring("this is a less boring func")
}
```

##### 进入正题
Venice 的 co 函数，传入的参数是一个函数，在 co 的内部会执行这个传入的函数，但是并不会等待这个函数执行结束，对于 co 的调用者来说，co 函数本身会立刻返回。co 函数其实是开启了一个新的协程 (轻量级线程) 来真正的执行传入的函数。

```
import Foundation
import Venice

private func boring(msg: String) {
    for i in 0..<Int.max {
        print("\(msg) \(i)")
        nap(for: Int(arc4random_uniform(1000) + 1).milliseconds)//sleep
    }
}

public func run03() {
    co {
        boring("co a less boring func")
    }
    
    /**
    //if do not want run03() finish, run the loop below
    for i in 0..<Int.max {
        yield
    }
    */
    print("run03() will return")
}
```

上面这段代码的运行结果如下

```
co a less boring func 0
run03() will return
```

可以看到，boring函数里面的循环只执行了一次，这是因为 co 函数是立刻返回的，紧接着，run03() 执行完 print 后也立刻返回，然后 run03() 的调用者 main 函数也就执行结束了 (进程结束)，之前 co 启动的协程自然也就无法继续执行了。

如果想让 co 里面的协程一直运行下去，可以在 co 调用返回后，执行代码中的那段 for loop。

*要注意的一点是，for loop 里面调用的 yield，是 Venice 引入的一种操作，意思是让出 CPU 给其他的协程。Golang 是不需要手动进行这种调用的，runtime 会自动的进行调度。*

*在 Venice 里面，如果是在 channel 上进行读写操作，读写的同时已经相当于调用过 yield 了，所以也不需要使用者再次显式的调用 yield。在后面的例子的，就会看到这种不需要手动调用 yield 的场景。*

##### 继续改动代码
调整代码成下面这个样子，在 co 调用后，让 run04() 所在的协程 sleep 一小段时间。

```
import Foundation
import Venice

private func boring(msg: String) {
    for i in 0..<Int.max {
        print("\(msg) \(i)")
        nap(for: Int(arc4random_uniform(1000) + 1).milliseconds)
    }
}

public func run04() {
    co(boring("co a less boring func"))
    print("I'm listening")
    nap(for: 2.second)
    print("You're boring; I'm leaving.")
}
```

这段代码的执行结果是下面这个样子的

```
co a less boring func 0
I'm listening
co a less boring func 1
co a less boring func 2
co a less boring func 3
co a less boring func 4
You're boring; I'm leaving.
```

*nap()是Venice提供的sleep函数，它的内部，相当于调用了yield。*

当main函数结束的时候，boring函数所在的协程也会结束。

##### 协程 (coroutine)
协程是一段独立运行的代码集合，通过 co 函数来启动。

协程的系统开销是很小的 (比 thread 小很多)，可以同时存在大量的协程 (具体到 Venice 底层使用的 [libmill](http://libmill.org/)，可以同时运行 **2000万个** 协程，并且每秒可以进行 **5000万次** 协程上下文切换)。

协程不是线程。

一个程序里面，可以只运行一个线程，但是在这个线程里面，可以包含千万个协程。

可以把协程看成是轻量级的线程。

##### 通讯 (communication) 
在 run04() 里面，是不能看到在协程中运行的 boring 函数的运行结果的。

boring 函数仅仅是把 msg 打印到了终端上。

想在协程之间真正的传递数据，需要用到通讯 (communication)。

##### Channel
在 Venice 里面，两个协程之间，通过 Channel\<Element\> 进行通讯。

Channel\<Element\> 的基本操作就是下面这3个:

```
	//声明、初始化
	let channel = Channel<String>()
```

```
	//在channel上发送数据
	channel.send("ping")
```

```
	//在channel上接收数据
	let message = channel.receive()!
```

##### 使用 Channel
用 channel 连接 boring 函数和 run05 函数

```
import Foundation
import Venice

private func boring(msg msg: String, channel: SendingChannel<String> ) {
    for i in 0..<Int.max {
        channel.send("\(msg) \(i)")
        nap(for: Int(arc4random_uniform(1000) + 1).milliseconds)
    }
}

public func run05() {
    let channel = Channel<String>()
    
    co {
        boring(msg: "co a less boring func", channel: channel.sendingChannel)
    }
    
    for _ in 0..<5 {
        print("You say: \(channel.receivingChannel.receive()!)")
    }
    
    print("You're boring; I'm leaving.")
    channel.close()
}
```

运行结果如下

```
You say: co a less boring func 0
You say: co a less boring func 1
You say: co a less boring func 2
You say: co a less boring func 3
You say: co a less boring func 4
You're boring; I'm leaving.
```

##### 同步 (Synchronization)
在 channel 上的读、写操作，是同步的、阻塞的。

run05() 执行到 channel.receivingChannel.receive()! 的时候，只有当 channel 里面有数据被写入的时候，这个读操作才会返回 (读到数据的时候才返回)，否则 run05() 就会一直在这里等待，不会继续往下执行。

同样的，在 boring 函数里面，执行 channel.send("\(msg) \(i)") 这个写操作的时候，只有当 channel 里面为空的时候，数据才能被写到 channel 里面，channel.send("\(msg) \(i)") 才会返回，否则，send 操作也会阻塞在这里。

在通讯过程中，发送者和接收者，必须都分别完成他们的写和读动作，否则双方就会一直互相等待下去 (死锁)。

channel 在协程之间完成通讯的同时，也达到了同步的目的。

##### 带缓冲的 channel
可以创建具有 buffer 的 channel。

这种 channel，当 buffer 还没有写满的时候，是没有前面描述的那种同步特性的。

buffering 有点类似 Erlang 语言里面的 mailboxes。

没有特殊理由的时候，不应该使用 buffered channel。

这篇文章后续的讨论，都不会使用 buffer。

##### Golang 哲学
Don't communicate by sharing memory, share memory by communicating.

### 模式 (Patterns)
##### Generator 模式：通过函数返回一个 channel 给调用者
Channel 是一等公民，和 class、struct、closure 同等重要。

```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            channel.send("\(msg) \(i)")
            nap(for: Int(arc4random_uniform(1000) + 1).milliseconds)
        }
    }
    
    return channel.receivingChannel
}

public func run06() {
    let receivingChannel = boring("co a less boring func")
    
    for _ in 0..<5 {
        print("You say: \(receivingChannel.receive()!)")
    }
    
    print("You're boring; I'm leaving.")
}
```

这段代码和前面的代码的运行结果，没有什么差别

```
You say: co a less boring func 0
You say: co a less boring func 1
You say: co a less boring func 2
You say: co a less boring func 3
You say: co a less boring func 4
You're boring; I'm leaving.
```

但是代码本身确有明显的变化，boring 函数返回一个 channel 给调用者，同时，在 boring 函数内部，通过 co 启动一个新的协程做具体的业务，并且通过刚才创建的 channel 把结果发送出去。

##### 利用 channel 作为 service 的接口
boring 函数对外提供了一个 service，这个 service 运行在独立的协程里面，并且通过channel 把数据传递给 service 的使用者。

可以同时使用多个 service。

```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            channel.send("\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)")
            nap(for: sleepTime)
        }
    }
    
    return channel.receivingChannel
}


public func run07() {
    let joe = boring("Joe")
    let ann = boring("Ann")
    
    for _ in 0..<5 {
        print("\(joe.receive()!)")
        print("\(ann.receive()!)")
    }
    
    print("You're both boring; I'm leaving.")
}
```

运行结果如下

```
Joe 0  (will sleep 996 ms)
Ann 0  (will sleep 681 ms)
Joe 1  (will sleep 173 ms)
Ann 1  (will sleep 147 ms)
Joe 2  (will sleep 750 ms)
Ann 2  (will sleep 374 ms)
Joe 3  (will sleep 318 ms)
Ann 3  (will sleep 705 ms)
Joe 4  (will sleep 126 ms)
Ann 4  (will sleep 828 ms)
You're both boring; I'm leaving.
```

##### 多路复用 (Multiplexing)
前面 run07() 里面的代码，始终都是先从 joe 里面读取数据，然后再从 ann 里面读取。如果 ann 里面的数据早于 joe 里面的数据就发送了，由于 channel 的同步特性，ann channel 其实会阻塞在它的 send 操作上，直到 run07 从 joe 里面读取完数据后，ann 所在的协程才能继续运行。

为了改善这种情况，可以使用 fan-in 模式。不管是 joe 还是 ann，只要有数据准备好并且执行了 send 操作，都可以立刻读取到。

```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            channel.send("\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)")
            nap(for: sleepTime)
        }
    }
    
    return channel.receivingChannel
}

private func fanIn(input1 input1: ReceivingChannel<String>, input2: ReceivingChannel<String>) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        while true {
            channel.send(input1.receive()!)
        }
    }
    
    co {
        while true {
            channel.send(input2.receive()!)
        }
    }
    
    return channel.receivingChannel
}

public func run08() {
    let joe = boring("Joe")
    let ann = boring("Ann")
    
    let c = fanIn(input1: joe, input2: ann)
    
    for _ in 0..<10 {
        print("\(c.receive()!)")
    }
    
    print("You're both boring; I'm leaving.")
}
```

运行结果如下

```
Joe 0  (will sleep 75 ms)
Ann 0  (will sleep 473 ms)
Joe 1  (will sleep 57 ms)
Joe 2  (will sleep 219 ms)
Joe 3  (will sleep 20 ms)
Joe 4  (will sleep 723 ms)
Ann 1  (will sleep 712 ms)
Joe 5  (will sleep 377 ms)
Ann 2  (will sleep 431 ms)
Joe 6  (will sleep 228 ms)
You're both boring; I'm leaving.
```

##### (Restoring sequencing)
前面 run08 里面的 fan-in 模式，boring 函数只负责 send 消息，并不需要消息的接收者做一个答复。如果需要，可以像下面这样修改代码

```
import Foundation
import Venice

private struct Message {
    let str: String
    let wait: Channel<Bool>
}

private let waitForIt = Channel<Bool>() // Shared between all messages

private func boring(msg: String) -> ReceivingChannel<Message> {
    let channel = Channel<Message>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            
            let message = Message(str: "\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)", wait: waitForIt)
            
            channel.send(message)
            nap(for: sleepTime)
            
            waitForIt.receive()!
        }
    }
    
    return channel.receivingChannel
}

private func fanIn(input1 input1: ReceivingChannel<Message>, input2: ReceivingChannel<Message>) -> ReceivingChannel<Message> {
    let channel = Channel<Message>()
    
    co {
        while true {
            channel.send(input1.receive()!)
        }
    }
    
    co {
        while true {
            channel.send(input2.receive()!)
        }
    }
    
    return channel.receivingChannel
}

public func run09() {
    let joe = boring("Joe")
    let ann = boring("Ann")
    
    let c = fanIn(input1: joe, input2: ann)
    
    for _ in 0..<5 {
        let message1 = c.receive()!
        print("\(message1.str)")
        message1.wait.send(true)
        
        let message2 = c.receive()!
        print("\(message2.str)")
        message2.wait.send(true)
    }
    
    print("You're both boring; I'm leaving.")
}
```

运行结果会是下面这个样子，并没有明显的区别

```
Joe 0  (will sleep 551 ms)
Ann 0  (will sleep 53 ms)
Ann 1  (will sleep 543 ms)
Joe 1  (will sleep 412 ms)
Ann 2  (will sleep 847 ms)
Joe 2  (will sleep 46 ms)
Joe 3  (will sleep 274 ms)
Joe 4  (will sleep 69 ms)
Joe 5  (will sleep 202 ms)
Ann 3  (will sleep 962 ms)
You're both boring; I'm leaving.
```

##### Select
前面介绍的多路复用技术，是通过启动多个协程实现的，每个 channel 对应一个协程。

另一种更常用的办法，是使用 select 操作，在一个协程里面同时读写多个 channel。

可以用 select 操作重新实现一遍 fan-in 模式

```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            channel.send("\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)")
            nap(for: sleepTime)
        }
        
    }
    
    return channel.receivingChannel
}

private func fanIn(input1 input1: ReceivingChannel<String>, input2: ReceivingChannel<String>) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        while true {
            select { when in
                when.receive(from: input1) { value in
                    //print("received \(value)")
                    channel.send(value)
                }
                
                when.receive(from: input2) { value in
                    channel.send(value)
                }
                
                when.otherwise {
                    //print("default case")
                }
            }
        }
    }
    
    return channel.receivingChannel
}

public func run10() {
    let joe = boring("Joe")
    let ann = boring("Ann")
    
    let c = fanIn(input1: joe, input2: ann)
    
    for _ in 0..<10 {
        print("\(c.receive()!)")
    }
    
    print("You're both boring; I'm leaving.")
}
```

运行结果和之前的 fan-in 没有区别

```
Ann 0  (will sleep 816 ms)
Joe 0  (will sleep 252 ms)
Joe 1  (will sleep 756 ms)
Ann 1  (will sleep 879 ms)
Joe 2  (will sleep 157 ms)
Joe 3  (will sleep 578 ms)
Ann 2  (will sleep 700 ms)
Joe 4  (will sleep 499 ms)
Joe 5  (will sleep 352 ms)
Ann 3  (will sleep 642 ms)
You're both boring; I'm leaving.
```

*这里用的 select 操作，和 Linux / Unix 里面的 select、poll、epoll，都是类似的，只不过前者监听的是 channel，后者监听的是 fd*

##### 在 Select 的基础上实现超时机制 (Timeout)
定时器是基于 channel 实现出来的，当达到定时时间的时候，定时器 channel 上会发送一个消息。

定时器可以放在 select 操作的里面


```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            nap(for: sleepTime)
            
            channel.send("\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)")
        }
    }
    
    return channel.receivingChannel
}


public func run11() {
    let joe = boring("Joe")
    
    var done = false
    while !done {
        select { when in
            when.receive(from: joe) { value in
                print("\(value)")
            }
            
            when.timeout(800.millisecond.fromNow()) {
                print("You are too slow.")
                done = true
            }
        }
    }
    
    print("You're boring; I'm leaving.")
}
```

运行结果是下面这个样子

```
Joe 0  (will sleep 48 ms)
Joe 1  (will sleep 706 ms)
Joe 2  (will sleep 747 ms)
Joe 3  (will sleep 304 ms)
You are too slow.
You're boring; I'm leaving.
```

##### Select 操作的整体超时
前面的 run11，是在每次进入 select 的时候，设置了一个超时 channel。

也可以在 while 循环的外面，设置一个整体的超时 channel，像下面这样

```
import Foundation
import Venice

private func boring(msg: String) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        for i in 0..<Int.max {
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            nap(for: sleepTime)
            
            channel.send("\(msg) \(i)  (will sleep \(Int(sleepTime * 1000)) ms)")
        }
    }
    
    return channel.receivingChannel
}

public func run12() {
    let joe = boring("Joe")
    let timeout = Timer(timingOut: 5.second.fromNow()).channel
    
    var done = false
    while !done {
        select { when in
            when.receive(from: joe) { value in
                print("\(value)")
            }
            
            when.receive(from: timeout) { _ in
                print("You are too slow.")
                done = true
            }
        }
    }
    
    print("You're boring; I'm leaving.")
}
```

运行结果如下

```
Joe 0  (will sleep 586 ms)
Joe 1  (will sleep 226 ms)
Joe 2  (will sleep 297 ms)
Joe 3  (will sleep 850 ms)
Joe 4  (will sleep 442 ms)
Joe 5  (will sleep 525 ms)
Joe 6  (will sleep 730 ms)
Joe 7  (will sleep 227 ms)
Joe 8  (will sleep 630 ms)
Joe 9  (will sleep 411 ms)
You are too slow.
You're boring; I'm leaving.
```

##### quit channel
boring 函数的调用者，可以主动的让 boring 内部的协程停止工作，也是通过 channel 来实现。

```
import Foundation
import Venice

private func boring(msg msg: String, quit: ReceivingChannel<Bool>) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        forSelect { when, done in
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            nap(for: sleepTime)
            
            when.send("\(msg), and will sleep \(Int(sleepTime * 1000)) ms", to: channel) {
                //print("sent value")
            }
            when.receive(from: quit) { _ in
                done()
            }
        }
        
        channel.close()
    }
    
    return channel.receivingChannel
}

public func run13() {
    let quit = Channel<Bool>()
    let joe = boring(msg: "Joe", quit: quit.receivingChannel)
    
    for _ in 0..<Int64(arc4random_uniform(10) + 1) {
        print("\(joe.receive()!)")
    }
    
    quit.send(true)
    
    print("You're boring; I'm leaving.")
}
```

运行结果仍然是类似的

```
Joe, and will sleep 154 ms
Joe, and will sleep 390 ms
Joe, and will sleep 133 ms
Joe, and will sleep 520 ms
Joe, and will sleep 752 ms
Joe, and will sleep 482 ms
Joe, and will sleep 47 ms
Joe, and will sleep 359 ms
You're boring; I'm leaving.
```

##### 在 quit channel 上接收消息
接着上面的例子，当 run13 向 quit channel 发送 true 的时候，run13 怎样才能知道 boring 函数成功的结束了自己的运行呢？让 boring 告诉它的调用者就行，同样，还是通过 quit channel。

```
import Foundation
import Venice

private func cleanup() {
    print("Here, do clean up")
}

private func boring(msg msg: String, quit: Channel<String>) -> ReceivingChannel<String> {
    let channel = Channel<String>()
    
    co {
        forSelect { when, done in
            let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
            nap(for: sleepTime)
            
            when.send("\(msg), and will sleep \(Int(sleepTime * 1000)) ms", to: channel) {
                //print("sent value")
            }
            when.receive(from: quit) { _ in
                cleanup()
                quit.send("See you!")
                done()
            }
        }
        
        channel.close()
    }
    
    return channel.receivingChannel
}

public func run14() {
    let quit = Channel<String>()
    let joe = boring(msg: "Joe", quit: quit)
    
    for _ in 0..<Int64(arc4random_uniform(10) + 1) {
        print("\(joe.receive()!)")
    }
    
    quit.send("Bye")
    print("Joe says: \(quit.receive()!)")
    
    print("You're boring; I'm leaving.")
}
```

现在运行结果会变成下面这个样子

```
Joe, and will sleep 220 ms
Joe, and will sleep 736 ms
Joe, and will sleep 308 ms
Joe, and will sleep 858 ms
Joe, and will sleep 527 ms
Joe, and will sleep 163 ms
Joe, and will sleep 844 ms
Here, do clean up
Joe says: See you!
You're boring; I'm leaving.
```

##### Daisy-chain

```
import Foundation
import Venice

private func f(left left: Channel<Int>, right: Channel<Int>) {
    left.send(right.receive()! + 1)
}

public func run15() {
    let leftMost = Channel<Int>()
    
    var right = leftMost
    var left = leftMost
    
    for _ in 0..<10000 {
        right = Channel<Int>()
        co {
            f(left: left, right: right)
        }
        left = right
    }
    
    co {
        right.send(1)
    }
    
    print("Joe says: \(leftMost.receive()!)")
    
    print("You're boring; I'm leaving.")
}
```

运行结果如下

```
Joe says: 10001
You're boring; I'm leaving.
```

### 系统软件 (Systems Software)
让我们具体看一下 CSP 这种并发模型，是如何用在系统软件的开发中的。

##### 例子：Google Search
问: Google search 需要做什么事情?


答: 输入一个搜索关键字 (query)，得到一组搜索结果 (和一些广告)。


问: 怎样获取这样的一组搜索结果？


答: 把搜索关键字分别发送给 Web search service，Image search service，YouTube search service，Maps search service，News search service 等等，然后把它们返回的结果再组合到一起。

那么，怎样做呢？

##### 模拟各种 search service
模拟 3 个 search service，每次执行 search 的时候，随机延时一小段时间。

```
import Foundation
import Venice

public typealias GoogleSearchResult = String

internal func fakeSearch(kind: String) -> (String) -> GoogleSearchResult {
    func search(query: String) -> GoogleSearchResult {
        let sleepTime = Int(arc4random_uniform(1000) + 1).milliseconds
        //print("-->\(kind) search use time: \(Int(sleepTime * 1000)) ms")
        nap(for: sleepTime)
        
        return GoogleSearchResult("\(kind) result for \(query), use time: \(Int(sleepTime * 1000)) ms")
    }
    
    return search
}

let web = fakeSearch("web")
let image = fakeSearch("image")
let video = fakeSearch("video")



//some util
internal func time(desc: String, function: ()->()) {
    let start : UInt64 = mach_absolute_time()
    function()
    let duration : UInt64 = mach_absolute_time() - start
    
    var info : mach_timebase_info = mach_timebase_info(numer: 0, denom: 0)
    mach_timebase_info(&info)
    
    let total = (duration * UInt64(info.numer) / UInt64(info.denom)) / NSEC_PER_MSEC
    print("\(desc)\(total) ms.")
}


protocol GoogleSearchResultDebugAble {
    func log()
}

extension GoogleSearchResult: GoogleSearchResultDebugAble {
    func log() {
        print("  \(self)")
    }
}

internal extension Array where Element: GoogleSearchResultDebugAble {
    internal func log() {
        print("google search result is:")
        for searchResult in self {
            searchResult.log()
        }
    }
}
```

##### Google Search 1.0
google 函数有一个输入参数，返回一个数组。

google 内部按照顺序依次调用 web、image、video search service，然后把它们的结果组装在一个数组内。

```
import Foundation
import Venice

private func google(query: String) -> Array<GoogleSearchResult> {
    var results = Array<GoogleSearchResult>()
    
    results.append(web(query))
    results.append(image(query))
    results.append(video(query))
    
    return results
}

public func run17() {
    var result: Array<GoogleSearchResult>?
    
    time("google search v1.0, use time: ") { () -> () in
        result = google("CSP")
    }
    
    result?.log()
}
```

运行结果是下面这个样子

```
google search v1.0, use time: 1237 ms.
google search result is:
  web result for CSP, use time: 743 ms
  image result for CSP, use time: 240 ms
  video result for CSP, use time: 243 ms
```

##### Google Search 2.0
并发调用 web、image、video search service，然后等待它们的返回结果。

不使用锁机制，不使用条件状态变量，不使用 callback。

```
import Foundation
import Venice

private func google(query: String) -> Array<GoogleSearchResult> {
    let channel = Channel<GoogleSearchResult>()
    
    co(channel.send(web(query)))
    co(channel.send(image(query)))
    co(channel.send(video(query)))
    
    var results = Array<GoogleSearchResult>()
    for _ in 0..<3 {
        results.append(channel.receive()!)
    }
    return results
}

public func run18() {
    var result: Array<GoogleSearchResult>?
    
    time("google search v2.0, use time: ") { () -> () in
        result = google("CSP")
    }
    
    result?.log()
}
```

运行结果如下

```
google search v2.0, use time: 871 ms.
google search result is:
  image result for CSP, use time: 40 ms
  video result for CSP, use time: 307 ms
  web result for CSP, use time: 864 ms
```

很明显，并发执行的效果比顺序执行的效果好很多。

##### Google Search 2.1
还可以加上超时机制，如果某个 search service 执行的时间太长，就不等待它的返回结果。

不使用锁机制，不使用条件状态变量，不使用 callback。

```
import Foundation
import Venice

private func google(query: String) -> Array<GoogleSearchResult> {
    let channel = Channel<GoogleSearchResult>()
    
    co(channel.send(web(query)))
    co(channel.send(image(query)))
    co(channel.send(video(query)))
    
    var results = Array<GoogleSearchResult>()
    
    let timeout = Timer(timingOut: 800.milliseconds.fromNow()).channel
    
    var done = false
    for _ in 0..<3 {
        if done == true {
            break
        }
        
        select { when in
            when.receive(from: channel) { value in
                results.append(value)
            }
            
            when.receive(from: timeout) { _ in
                print("timeout.")
                done = true
            }
        }
    }
    return results
}

public func run19() {
    var result: Array<GoogleSearchResult>?
    
    time("google search v2.1, use time: ") { () -> () in
        result = google("CSP")
    }
    
    result?.log()
}
```

如果看到下面这种形式的运行结果，则说明是触发了超时的条件

```
timeout.
google search v2.1, use time: 810 ms.
google search result is:
  web result for CSP, use time: 341 ms
  video result for CSP, use time: 537 ms
```

##### 避免超时
问：怎样才能避免丢弃响应速度更慢的服务器返回的搜索结果？

答：使用 Replicate 策略。同时向多个同类型的 search service 发送请求，使用第一个返回来的查询结果。

```
private func first(query query: String, replicas: ((String) -> GoogleSearchResult)...) -> GoogleSearchResult {
    let channel = Channel<GoogleSearchResult>()
    
    for search in replicas {
        co(channel.send(search(query)))
    }
    
    return channel.receive()!
}
```

##### Google Search 3.0
仍然不使用锁机制，不使用条件状态变量，不使用 callback。

```
import Foundation
import Venice


let web1 = fakeSearch("web1")
let web2 = fakeSearch("web2")
let image1 = fakeSearch("image1")
let image2 = fakeSearch("image2")
let video1 = fakeSearch("video1")
let video2 = fakeSearch("video2")


private func first(query query: String, replicas: ((String) -> GoogleSearchResult)...) -> GoogleSearchResult {
    let channel = Channel<GoogleSearchResult>()
    
    for search in replicas {
        co(channel.send(search(query)))
    }
    
    return channel.receive()!
}


private func google(query: String) -> Array<GoogleSearchResult> {
    let channel = Channel<GoogleSearchResult>()
    
    co {
        channel.send(first(query: query, replicas: web1, web2))
    }
    
    co {
        channel.send(first(query: query, replicas: image1, image2))
    }
    
    co {
        channel.send(first(query: query, replicas: video1, video2))
    }
    
    var results = Array<GoogleSearchResult>()
    
    let timeout = Timer(timingOut: 1000.milliseconds.fromNow()).channel
    
    var done = false
    for _ in 0..<3 {
        if done == true {
            break
        }
        
        select { when in
            when.receive(from: channel) { value in
                //print("receive \(value)")
                results.append(value)
            }
            
            when.receive(from: timeout) { _ in
                print("timeout.")
                done = true
            }
        }
    }
    return results
}


public func run20() {
    var result: Array<GoogleSearchResult>?
    
    time("google search v3.0, use time: ") { () -> () in
        result = google("CSP")
    }
    
    result?.log()
}
```

最终的运行结果如下

```
google search v3.0, use time: 506 ms.
google search result is:
  web1 result for CSP, use time: 433 ms
  image1 result for CSP, use time: 434 ms
  video2 result for CSP, use time: 499 ms
```

### 不要过度使用
coroutine 和 channel 是一种很好的设计思想，可以解决某些类型的问题。

但是，有时我们仍然会面对一些需要用传统思路来解决的小问题，也就是基于锁机制 (共享内存)。

这两种不同的技术思路，并不冲突，它们是可以共存的。

正确的工具做正确的事情。

### 后记
这篇文章里面的 demo code 位于 [https://github.com/fengjian0106/CSP-tutorial.git](https://github.com/fengjian0106/CSP-tutorial.git)
