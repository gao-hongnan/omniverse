# Asynchronous Programming

## Intuition And Analogy

From the
[explain coroutines like I'm five](https://dev.to/thibmaek/explain-coroutines-like-im-five-2d9),
we see a good analogy below:

> You start watching the cartoon, but it's the intro. Instead of watching the
> intro you switch to the game and enter the online lobby - but it needs 3
> players and only you and your sister are in it. Instead of waiting for another
> player to join you switch to your homework, and answer the first question. The
> second question has a link to a YouTube video you need to watch. You open it -
> and it starts loading. Instead of waiting for it to load, you switch back to
> the cartoon. The intro is over, so you can watch. Now there are commercials -
> but meanwhile a third player has joined so you switch to the game And so on...

The idea is that you don't just switch the tasks really fast to make it look
like you are doing everything at once. You utilize the time you are waiting for
something to happen(IO) to do other things that do require your direct
attention.

So in a way the key idea is **_non-blocking_** and **_utilizing waiting time_**.

## References

-   [Next-Level Concurrent Programming In Python With Asyncio - ArjanCodes](https://www.youtube.com/watch?v=GpqAQxH1Afc&t=276s)
-   https://github.com/ArjanCodes/2022-asyncio/blob/main/asyncio_iter.py
-   https://realpython.com/async-io-python/
-   https://stackoverflow.com/questions/553704/what-is-a-coroutine
-   https://dev.to/thibmaek/explain-coroutines-like-im-five-2d9
