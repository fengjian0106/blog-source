title: iOS cache系统和自定义http代理
date: 2014-10-08 17:11:38
tags: [iOS, cache, proxy]
---

面试的时候，我喜欢问一些关于iOS http cache的问题，遇见不少开发者，第一反应都是自己编码实现整个cache存取流程，思路是没有问题的，但是这并不是我想要的答案。对于大部分使用场景，iOS自带的cache系统，就已经足够了，官方文档里也描述的很清楚，可以直接看 [Understanding Cache Access](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/URLLoadingSystem/Concepts/CachePolicies.html)。

### 流媒体播放的缓存需求
在做一个音乐播放器app的过程中，有一个需求是可以边播放边缓存，并且不能浪费流量(不能播放时有一次http下载，离线缓存时又有另一次http下载)。

### 技术方案
查了一大堆文档，做了很多原型代码，最终还是发现 `AVQueuePlayer` 并不是基于 [URL Loading System](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/URLLoadingSystem/URLLoadingSystem.html) 实现的，所以不可能直接使用其中的cache系统，只能想其他的办法，比如:

1. 使用 [CFNetwork](https://developer.apple.com/library/ios/documentation/Networking/Conceptual/CFNetwork/Introduction/Introduction.html)、[Audio File Stream Services](https://developer.apple.com/library/ios/documentation/MusicAudio/Reference/AudioStreamReference/index.html) 以及 [Audio Queue Services](https://developer.apple.com/library/ios/documentation/MusicAudio/Conceptual/AudioQueueProgrammingGuide/Introduction/Introduction.html) 分别实现http下载、多媒体解码和播放过程，因为所有的流程都是自己控制，自然也就可以实现边播边存的效果。  
2. 使用 `ffmpeg` 第三方多媒体框架来实现。  
3. 仍然使用iOS自带的 `AVQueuePlayer` 进行http流媒体播放，但是在app内运行一个支持cache的 `http proxy`，让 `AVQueuePlayer` 通过这个 proxy 请求多媒体文件。

### 方案对比
第一种方案是比较容易想到的，但是其实实现的难度比较大，三个模块都要使用偏底层的接口，学习成本很高。第二种方案其实也是学习成本很高，而且只能使用 `ffmpeg` 的软解码器，功耗也会很大。前两个方案还有一个共同的问题，就是需要自己考虑如何和iOS系统做整合，比如如何优雅的实现后台播放功能等等。综合考虑后，还是选择了第三种方案，虽然也有一定的学习成本，而且需要修改标准的 `http proxy` 协议，但是大部分模块都是用iOS的SDK实现的，和操作系统贴合的最紧密。

### 实现 http proxy
自己完整的实现一个 `http proxy`，也是一件很复杂的事情，所以还是偏向于找开源方案，对比了几个开源方案后，最终选择在 [Polipo](http://www.pps.univ-paris-diderot.fr/~jch/software/polipo/) 的基础上进行修改。
  
有一个技术细节需要说明一下，当使用浏览器、不使用代理的时候，通过抓包可以看到浏览器发出的http请求如下(忽略了无关内容):  

```   
GET / HTTP/1.1
Host: www.example.com
Connection: keep-alive
```

当使用浏览器并且使用代理的时候，通过抓包可以看到浏览器发出的http请求如下(忽略了无关内容):

```   
GET http://www.example.com/ HTTP/1.1
Host: www.example.com
Proxy-Connection: keep-alive
```

最重要的一个区别就是 `GET` 请求后面的路径信息，前者是相对路径 `/`，而后者是绝对路径 `http://www.example.com/`。因为浏览器本身支持设置代理，所以浏览器会拼接合适的路径信息并且发送。但是iOS的 `AVQueuePlayer` 并不支持 `http proxy` 功能，无法和标准的代理服务器协同工作，所以只能同时在 `AVQueuePlayer` 和 [Polipo](http://www.pps.univ-paris-diderot.fr/~jch/software/polipo/) 上做一些小的修改。

使用 `AVQueuePlayer` 的时候，需要做一些 `magic trick`，关键代码如下:

``` objectivec
httpProxyUrl = [NSURL URLWithString:[NSString stringWithFormat:@"http://127.0.0.1:%d/http://%@/music_new/%@", httpProxyPort, kKYMediaServiceManagerRemoteServerIpAndPort, mediaInfo.fileName]];
```

可以这样来理解这段代码，假设app内的 `http proxy` 的地址为 `127.0.0.1:9258`，实际的多媒体文件的地址为 `http://media.test.com/xxx.mp3`，那么 `AVQueuePlayer` 请求的最终地址就应该是:

```
http://127.0.0.1:9258/http://media.test.com/xxx.mp3
```

在 [Polipo](http://www.pps.univ-paris-diderot.fr/~jch/software/polipo/) 中，解析得到的最终目的服务器的地址(相对路径)是 `/http://media.test.com/xxx.mp3`，需要稍微修改一下源代码，去掉最左侧的 `/` 字符。具体就是在 `client.c` 文件的 `httpClientHandlerHeaders` 函数中添加一小段代码，也就是 `#ifdef POLIPO_KUYQI_VERSION` 和 `#endif` 之间的那一段:

```  objectivec
int
httpClientHandlerHeaders(FdEventHandlerPtr event, StreamRequestPtr srequest,
                         HTTPConnectionPtr connection)
{
    HTTPRequestPtr request;
    int rc;
    int method, version;
    AtomPtr url = NULL;
    int start;
    int code;
    AtomPtr message;

    start = 0;
    /* Work around clients working around NCSA lossage. */
    if(connection->reqbuf[0] == '\n')
        start = 1;
    else if(connection->reqbuf[0] == '\r' && connection->reqbuf[1] == '\n')
        start = 2;

    httpSetTimeout(connection, -1);
    
    
#ifdef POLIPO_KUYQI_VERSION
    char *pch;
    pch = strstr (connection->reqbuf, "/http://");//TODO: if the client use url encoding??
    if (pch == NULL) {
        //fprintf(stderr, "###########this is normal http request\r\n");
    } else {
        //remove the first '/'
        pch = strstr (connection->reqbuf, "/");
        pch[0] = ' ';
    }
#endif
    
    
    
    rc = httpParseClientFirstLine(connection->reqbuf, start,
                                  &method, &url, &version);
    if(rc <= 0) {
        do_log(L_ERROR, "Couldn't parse client's request line\n");
        code = 400;
        message =  internAtom("Error in request line");
        goto fail;
    }

    do_log(D_CLIENT_REQ, "Client request: ");
    do_log_n(D_CLIENT_REQ, connection->reqbuf, rc - 1);
    do_log(D_CLIENT_REQ, "\n");

    if(version != HTTP_10 && version != HTTP_11) {
        do_log(L_ERROR, "Unknown client HTTP version\n");
        code = 400;
        message = internAtom("Error in first request line");
        goto fail;
    }

    if(method == METHOD_UNKNOWN) {
        code = 501;
        message =  internAtom("Method not implemented");
        goto fail;
    }

    request = httpMakeRequest();
    if(request == NULL) {
        do_log(L_ERROR, "Couldn't allocate client request.\n");
        code = 500;
        message = internAtom("Couldn't allocate client request");
        goto fail;
    }

    if(connection->version != HTTP_UNKNOWN && version != connection->version) {
        do_log(L_WARN, "Client version changed!\n");
    }

    connection->version = version;
    request->flags = REQUEST_PERSISTENT;
    request->method = method;
    request->cache_control = no_cache_control;
    httpQueueRequest(connection, request);
    connection->reqbegin = rc;
    return httpClientRequest(request, url);

 fail:
    if(url) releaseAtom(url);
    shutdown(connection->fd, 0);
    connection->reqlen = 0;
    connection->reqbegin = 0;
    httpConnectionDestroyReqbuf(connection);
    connection->flags &= ~CONN_READER;
    httpClientNewError(connection, METHOD_UNKNOWN, 0, code, message);
    return 1;

}
```


