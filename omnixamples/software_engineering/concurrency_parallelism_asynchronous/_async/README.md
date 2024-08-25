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

So in a way the key idea is **switching**, **_non-blocking_** and **_utilizing
waiting time_**.

---

Absolutely! Let's delve into the conceptual workings of asynchronous programming
in Python, particularly focusing on how `asyncio` works under the hood. We'll
then use an analogy to help solidify the understanding.

### Deep Conceptual Explanation

#### Event Loop

The core of `asyncio` is the event loop, which is responsible for managing and
scheduling asynchronous tasks. Here’s how it works step-by-step:

1. **Task Management**: The event loop keeps track of all the tasks that need to
   run. Each task represents an asynchronous operation.
2. **Scheduling**: Tasks are not run immediately but are scheduled to run when
   the conditions are right (e.g., an I/O operation can proceed without
   blocking).
3. **Non-blocking Execution**: The event loop continuously checks if tasks are
   ready to run (for example, whether a non-blocking I/O operation can be
   started or has completed). It switches between tasks, executing them in small
   chunks.

#### Coroutines and Tasks

Coroutines are the fundamental units of asynchronous code in Python, defined
using `async def`. Here’s their role:

-   When you define a coroutine, it is merely a blueprint. When a coroutine is
    called, it doesn’t run by itself. Instead, it needs to be scheduled into the
    event loop.
-   When you use `await` on a coroutine, you are effectively yielding control
    back to the event loop, saying, “I can’t proceed until some external
    condition (like an I/O operation) is completed. Please run other tasks in
    the meantime.”

#### Concurrency with `asyncio.gather()`

`asyncio.gather()` is used for managing multiple coroutines concurrently:

-   It takes multiple awaitable objects (coroutines, tasks, or futures).
-   It schedules them to run concurrently, which means the event loop starts
    their execution without waiting for each one to finish before starting the
    next.
-   It waits for all of them to complete, aggregating their results.

### Under the Hood Mechanics

When `asyncio.gather()` is called with multiple coroutines:

1. **Initialization**: Each coroutine is initialized and prepared to run.
2. **Task Wrapping**: Internally, each coroutine is wrapped into a Task. A Task
   is a special object that manages the state and execution of a coroutine.
3. **Execution**: The event loop begins executing these tasks. If any coroutine
   performs an `await`, the Task pauses that coroutine and gives control back to
   the event loop.
4. **Context Switching**: The event loop efficiently switches between tasks
   whenever one is paused (awaiting). It uses this time to progress other tasks
   that are ready to run.
5. **Completion**: Once the external condition (like an I/O completion) of a
   paused task is met, it resumes. This continues until all tasks are complete.

### Analogy: Restaurant Operations

Imagine a chef (the event loop) in a busy restaurant kitchen (the program):

-   **Orders**: Each dish to be prepared (task or coroutine) is an order placed
    by customers.
-   **Concurrent Preparation**: The chef can start multiple dishes at once.
    While waiting for one dish to simmer (awaiting an I/O operation), the chef
    starts chopping vegetables for another dish.
-   **Interruptions**: If the chef realizes they need an ingredient from the
    pantry (awaiting an external event), they ask an assistant to fetch it and
    continue with other dishes in the meantime.
-   **Efficient Kitchen**: The kitchen (event loop) is most efficient when the
    chef keeps moving, switching tasks based on what can be progressed
    immediately.

This analogy shows how `asyncio` manages tasks—keeping busy and productive by
switching tasks based on readiness and external conditions, ensuring that time
is utilized effectively without idle waiting.

By understanding these mechanics and the restaurant analogy, you can better
appreciate how asynchronous programming in Python allows for efficient handling
of I/O-bound tasks, maximizing productivity (like our chef) and responsiveness.

#### **Setting**: A Busy Restaurant Kitchen

The chef (event loop) is tasked with preparing a large dinner order consisting
of various dishes that require different preparation and cooking times.

#### **Tasks and Their Specific Actions**

1. **Soup (Network Request)**:

    - **Active Preparation**: Chopping vegetables and preparing stock.
    - **Waiting Time (Await)**: Simmering the soup.
    - **Resumption**: Adding final seasonings and adjusting the temperature
      before serving.

2. **Roast (Database Query)**:

    - **Active Preparation**: Seasoning the meat and preheating the oven.
    - **Waiting Time (Await)**: Roasting in the oven.
    - **Resumption**: Checking the roast's temperature, letting it rest, and
      then carving it for serving.

3. **Salad (File I/O)**:

    - **Active Preparation**: Washing and chopping vegetables, preparing
      dressing.
    - **Waiting Time (Await)**: Letting the salad dressing meld with the
      vegetables in the fridge.
    - **Resumption**: Adding final touches like nuts or croutons before serving.

4. **Dessert (User Input)**:
    - **Active Preparation**: Mixing ingredients for a cake and prepping the
      baking tin.
    - **Waiting Time (Await)**: Baking the cake in the oven.
    - **Resumption**: Decorating the cake once it has cooled down.

#### **Simultaneous Task Management**

-   **Starting the Soup and Roast**: The chef starts by preparing the soup and
    roast since they have the longest cooking times. While the soup is left to
    simmer and the roast to cook in the oven, the chef doesn’t wait idly.

-   **Salad and Dessert Preparation**: With the soup and roast cooking, the chef
    begins preparing the salad and dessert. The chef mixes the salad and places
    it in the fridge, then starts mixing the dessert batter.

-   **Efficient Switching**: As the dessert goes into the oven, the chef checks
    on the roast. With the roast still cooking, the chef returns to the soup to
    add the final seasonings. Each switch is made at an optimal time to ensure
    no task is left unattended longer than necessary.

#### **Efficient Use of Waiting Times**

-   **Assistant Tasks (API Calls)**: While handling these main tasks, the chef
    asks an assistant to fetch additional ingredients from the pantry, similar
    to making asynchronous API calls. The chef continues with other
    preparations, like setting the tables or garnishing dishes, maximizing the
    use of time.

#### **Completing the Tasks**

-   **Sequential Finishing**: As each dish completes its cooking or waiting
    time, the chef finalizes them. The salad is taken out, tossed with the final
    ingredients, and the dessert is decorated. Everything is timed to ensure
    dishes are ready to serve hot and fresh, maximizing the quality of the
    output.

## References

-   [Next-Level Concurrent Programming In Python With Asyncio - ArjanCodes](https://www.youtube.com/watch?v=GpqAQxH1Afc&t=276s)
-   https://github.com/ArjanCodes/2022-asyncio/blob/main/asyncio_iter.py
-   https://realpython.com/async-io-python/
-   https://stackoverflow.com/questions/553704/what-is-a-coroutine
-   https://dev.to/thibmaek/explain-coroutines-like-im-five-2d9
