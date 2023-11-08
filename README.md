# Omniverse

![Python version](https://img.shields.io/badge/Python-3.9-3776AB)
[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)

🌌 Omniverse: A cosmic collection of machine learning, deep learning, data
science, math, and software engineering explorations. Dive into the universe of
knowledge! 🚀

## Building the book

To use a custom domain with GitHub Pages and with Jupyter Book, we would need to
follow the instructions given
[here](https://jupyterbook.org/en/stable/publish/gh-pages.html#use-a-custom-domain-with-github-pages).

1. **Add Custom Domain to GitHub Pages Settings**:

   - Go to your GitHub repository.
   - Click on "Settings".
   - Scroll down to the "GitHub Pages" section.
   - In the "Custom domain" box, enter your custom domain (e.g.,
     `gaohongnan.com`) and save.
   - You might see the "improperly configured" error, which is expected at this
     stage since the DNS hasn't been set up yet.

   > Make sure you add your custom domain to your GitHub Pages site before
   > configuring your custom domain with your DNS provider. Configuring your
   > custom domain with your DNS provider without adding your custom domain to
   > GitHub could result in someone else being able to host a site on one of
   > your subdomains. From GitHub
   > [documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#about-custom-domain-configuration)

2. **Modify DNS Settings at Domain Registrar**:

   - Head over to your domain registrar.
   - Configure the DNS settings:
     - For an apex domain: Set up the **A records**.
     - For a `www` subdomain: Set up the **CNAME record** pointing to your
       GitHub Pages URL.

3. **Wait and Check**:

   - Now, you'll need to wait for DNS propagation. This can take some time.
   - After a while (it could be a few hours), return to your GitHub Pages
     settings. The error should resolve itself once the DNS has fully propagated
     and GitHub can detect the correct settings.

4. **Enforce HTTPS**:
   - Once the error is gone, you can then opt to "Enforce HTTPS" for added
     security.

In essence, you temporarily accept the error message in your GitHub Pages
settings after adding the custom domain. After you've configured the DNS
settings at your domain registrar and they've propagated, the error in GitHub
Pages settings should clear up.

The main goal of GitHub's recommendation is to make sure you've shown intent to
use the domain with GitHub Pages before setting it up with your DNS provider, to
prevent potential subdomain takeovers. By adding the custom domain in the
repository settings (even if it throws an error initially), you've asserted this
intent.

## How to Index Jupyter Book?

- [Indexing on search engines](https://github.com/executablebooks/jupyter-book/issues/1934)
- [Generate sitemap.xml for SEO](https://github.com/executablebooks/jupyter-book/issues/880)
