# Characteristic Function

## Why Characteristic Function?

Earlier in the Moment Generating Functions, we noted that the MGF is not defined for all random variables. The MGF is not defined if the distribution does not have **finite moments**.

For example, the Cauchy distribution has no moments of any order. Let $X$ be a Cauchy random variable with density function

$$
\mathrm{f}_{\mathrm{X}}(\mathrm{x})=\frac{\frac{1}{\pi}}{1+\mathrm{x}^2}, \quad \text { for all } \mathrm{x} \in \mathbb{R}
$$

You can show that for any nonzero real number $\mathrm{s}$ {cite}`pishro-nik_2014`,

$$
M_X(s)=\int_{-\infty}^{\infty} e^{s x} \frac{\frac{1}{\pi}}{1+x^2} d x=\infty .
$$

Therefore, the moment generating function does not exist for this random variable on any real interval with positive length. If a random variable does not have a well-defined MGF, we can use the characteristic function defined as

$$
\Phi_{\mathrm{X}}(\omega)=\mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega \mathrm{X}}\right],
$$

where $\mathrm{j}=\sqrt{-1}$ and $\omega$ is a real number. It is worth noting that $\mathrm{e}^{\mathrm{j} \omega \mathrm{X}}$ is a complex-valued random variable. We have not discussed complex-valued random variables. Nevertheless, you can imagine that a complex random variable can be written as $\mathrm{X}=\mathrm{Y}+\mathrm{jZ}$, where $\mathrm{Y}$ and $\mathrm{Z}$ are ordinary realvalued random variables. Thus, working with a complex random variable is like working with two real-valued random variables. The advantage of the characteristic function is that it is defined for all real-valued random variables. Specifically, if $X$ is a real-valued random variable, we can write

$$
\left|e^{j \omega X}\right|=1 \text {. }
$$

Therefore, we conclude

$$
\begin{aligned}
\left|\Phi_{\mathrm{X}}(\omega)\right| & =\left|\mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega \mathrm{X}}\right]\right| \\
& \leq \mathbb{E}\left[\left|\mathrm{e}^{\mathrm{j} \omega \mathrm{X}}\right|\right] \\
& \leq 1 .
\end{aligned}
$$

The characteristic function has similar properties to the MGF. For example, if $\mathrm{X}$ and $\mathrm{Y}$ are independent

$$
\begin{aligned}
& \Phi_{\mathrm{X}+\mathrm{Y}}(\omega)=\mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega(\mathrm{X}+\mathrm{Y})}\right] \\
& =\mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega \mathrm{X}} \mathrm{e}^{\mathrm{j} \omega \mathrm{Y}}\right] \\
& =\mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega \mathrm{X}}\right] \mathbb{E}\left[\mathrm{e}^{\mathrm{j} \omega \mathrm{Y}}\right] \quad \text { (since } \mathrm{X} \text { and } \mathrm{Y} \text { are independent) } \\
& =\Phi_X(\omega) \Phi_Y(\omega) \text {. } \\
&
\end{aligned}
$$

More generally, if $X_1, X_2, \ldots, X_n$ are $\mathrm{n}$ independent random variables, then

$$
\Phi_{X_1+X_2+\cdots+X_n}(\omega)=\Phi_{X_1}(\omega) \Phi_{X_2}(\omega) \cdots \Phi_{X_n}(\omega) .
$$

*Section content is based on [Introduction to probability, statistics, and Random Processes](https://www.probabilitycourse.com/chapter6/6_1_4_characteristic_functions.php).*

## Characteristic Function and Fourier Transform

See {cite}`chan_2021` section 6.1.3 for a detailed discussion.

## Further Readings

- Chan, Stanley H. "Chapter 6.1.3. Characteristic Functions." In Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
- Pishro-Nik, Hossein. "Chapter 6.1.4. Characteristic Functions." In Introduction to Probability, Statistics, and Random Processes. Kappa Research, 2014.