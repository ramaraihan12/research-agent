# What does HTTP status code 418 mean, and which RFC defines it?

HTTP status code 418, known as "I'm a teapot," indicates that a server refuses to brew coffee because it is permanently a teapot [1][2]. This code was originally defined in RFC 2324, published on April 1, 1998, as an April Fools' Day RFC [1][2]. Although RFC 9110 later marked the code as unused, public outcry led to it being reserved instead [1][2].

*   **Original Definition**: The code stems from the Hyper Text Coffee Pot Control Protocol (HTCPCP), a satirical protocol where a teapot returns 418 when requested to brew coffee [1][2].
*   **Extensions**: RFC 7168, also an April Fools' RFC from 2014, extended HTCPCP to support brewing teas [1].
*   **Usage**: While intended as a joke, some websites now use 418 for requests they do not wish to handle, such as automated queries [2].
*   **Distinction**: If a combined coffee/tea pot is temporarily out of coffee, it should return status 503 rather than 418 [2].

The sources do not settle on a single definitive reason for the code's survival, as they present conflicting narratives regarding whether its retention was purely due to the "Save 418 Movement" or if it was formally reserved to prevent non-joke semantics despite being marked unused.

## Sources

1. [Hyper Text Coffee Pot Control Protocol - Wikipedia](https://en.wikipedia.org/wiki/Hyper_Text_Coffee_Pot_Control_Protocol)
2. [418 I'm a teapot - HTTP | MDN - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/418)
