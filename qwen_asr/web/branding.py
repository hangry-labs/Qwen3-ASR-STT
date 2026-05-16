from __future__ import annotations

import html


def brand_css(asset_base: str = "/assets/hangrylabs") -> str:
    return f"""
:root {{
    --brand-cream: #fff3e7;
    --brand-muted: rgba(255, 243, 231, 0.72);
}}

.gradio-container {{
    padding-top: 0 !important;
}}

.gradio-container > .main {{
    padding-top: 0 !important;
}}

.brand-hero {{
    overflow: hidden;
    position: relative;
    width: calc(100% + 24px);
    min-height: 330px;
    box-sizing: border-box;
    margin: 0 -12px 18px;
    padding: 22px;
    border: 1px solid rgba(255, 176, 118, 0.24);
    border-radius: 16px;
    background:
        linear-gradient(115deg, rgba(8, 5, 3, 0.96) 0%, rgba(20, 9, 2, 0.86) 48%, rgba(255, 107, 0, 0.24) 100%),
        url("{asset_base}/banner.jpg") center / cover no-repeat;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.58), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}}

.brand-nav {{
    position: relative;
    z-index: 1;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 42px;
}}

.brand-lockup {{
    display: flex;
    align-items: center;
    gap: 18px;
    min-width: 0;
}}

.brand-logo-wrap img {{
    display: block;
    width: 96px;
    height: 96px;
    object-fit: contain;
    border-radius: 18px;
    box-shadow: 0 0 0 1px rgba(255, 176, 118, 0.24), 0 18px 34px rgba(0, 0, 0, 0.42);
}}

.brand-name {{
    color: var(--brand-cream);
    font-size: 1.02rem;
    font-weight: 800;
    line-height: 1.08;
}}

.brand-subname {{
    margin-top: 3px;
    color: var(--brand-muted);
    font-size: 0.82rem;
}}

.brand-links {{
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 0px;
    margin-right: 90px;
}}

.brand-links a {{
    display: inline-flex;
    align-items: center;
    position: relative;
    overflow: hidden;
    min-height: 34px;
    padding: 7px 12px;
    border: 1px solid rgba(255, 176, 118, 0.28);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.065);
    color: var(--brand-cream) !important;
    font-size: 0.84rem;
    font-weight: 700;
    text-decoration: none !important;
    backdrop-filter: blur(8px);
    transform: translateY(0);
    transition: transform 150ms ease, border-color 150ms ease, background 150ms ease, box-shadow 150ms ease;
}}

.brand-links a:hover {{
    transform: translateY(-2px);
    border-color: rgba(255, 176, 118, 0.55);
    background: rgba(255, 255, 255, 0.11);
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.28);
}}

.brand-links a:first-child {{
    border-color: rgba(255, 107, 0, 0.72);
    background: linear-gradient(135deg, #ff6b00, #ff9a3d);
    color: #180a02 !important;
    box-shadow: 0 0 0 rgba(255, 107, 0, 0.0);
    animation: examples-pulse 2.8s ease-in-out infinite;
}}

.brand-links a:first-child::after {{
    content: "";
    position: absolute;
    inset: -45% -70%;
    background: linear-gradient(115deg, transparent 42%, rgba(255, 255, 255, 0.58) 50%, transparent 58%);
    transform: translateX(-90%);
    animation: examples-shine 3.8s ease-in-out infinite;
    pointer-events: none;
}}

.brand-links a:first-child:hover {{
    background: linear-gradient(135deg, #ff7a12, #ffb066);
    box-shadow: 0 14px 30px rgba(255, 107, 0, 0.30);
}}

@keyframes examples-pulse {{
    0%, 100% {{
        box-shadow: 0 0 0 rgba(255, 107, 0, 0.0);
    }}
    50% {{
        box-shadow: 0 0 22px rgba(255, 107, 0, 0.34);
    }}
}}

@keyframes examples-shine {{
    0%, 58% {{
        transform: translateX(-90%);
    }}
    72%, 100% {{
        transform: translateX(90%);
    }}
}}

.brand-copy {{
    position: relative;
    z-index: 1;
    max-width: 760px;
}}

.brand-copy h1 {{
    margin: 0 0 12px;
    color: var(--brand-cream);
    font-size: clamp(2.4rem, 6vw, 5.15rem);
    line-height: 0.92;
}}

.brand-copy p {{
    max-width: 660px;
    margin: 0;
    color: var(--brand-muted);
    font-size: 1.04rem;
    line-height: 1.6;
}}

.brand-pills {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 24px;
}}

.brand-pills span {{
    position: relative;
    padding: 7px 11px;
    padding-left: 27px;
    border: 1px solid rgba(255, 176, 118, 0.24);
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(255, 107, 0, 0.16), rgba(0, 0, 0, 0.34));
    color: var(--brand-cream);
    font-size: 0.82rem;
    font-weight: 750;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 8px 18px rgba(0, 0, 0, 0.22);
}}

.brand-pills span::before {{
    content: "";
    position: absolute;
    left: 11px;
    top: 50%;
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(135deg, #ff6b00, #ffb076);
    box-shadow: 0 0 10px rgba(255, 107, 0, 0.45);
    transform: translateY(-50%);
}}

#build-badge {{
    position: absolute;
    right: 120px;
    bottom: 10px;
    z-index: 3;
    min-width: max-content;
    background: rgba(8, 5, 3, 0.76);
    color: var(--brand-cream);
    border: 1px solid rgba(255, 176, 118, 0.24);
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 10.5px;
    line-height: 1.35;
    font-family: Arial, sans-serif;
    text-align: left;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
}}

@media (max-width: 760px) {{
    .brand-hero {{
        width: 100%;
        min-height: 0;
        margin: 0 0 18px;
        padding: 16px;
        border-radius: 12px;
    }}

    .brand-nav {{
        align-items: flex-start;
        flex-direction: column;
        margin-bottom: 30px;
    }}

    .brand-links {{
        justify-content: flex-start;
        padding-top: 0;
        margin-right: 0;
    }}

    .brand-copy h1 {{
        font-size: 2.45rem;
    }}

    #build-badge {{
        position: relative;
        top: auto;
        right: auto;
        display: inline-block;
        min-width: 0;
        margin-top: 14px;
    }}
}}
"""


def brand_header_html(
    product_name: str,
    description: str,
    links: list[tuple[str, str]],
    capabilities: list[str],
    runtime_html: str,
    asset_base: str = "/assets/hangrylabs",
    lab_name: str = "Hangry Labs",
    lab_subtitle: str = "Local speech tools",
) -> str:
    rendered_links = "\n".join(
        f'            <a href="{html.escape(url, quote=True)}" target="_blank" rel="noreferrer">{html.escape(label)}</a>'
        for label, url in links
    )
    rendered_capabilities = "\n".join(f"        <span>{html.escape(value)}</span>" for value in capabilities)
    return f"""
<section class="brand-hero">
    <div class="brand-nav">
        <div class="brand-lockup">
            <div class="brand-logo-wrap">
                <img src="{html.escape(asset_base, quote=True)}/logo_small.png" alt="{html.escape(lab_name)} logo">
            </div>
            <div>
                <div class="brand-name">{html.escape(lab_name)}</div>
                <div class="brand-subname">{html.escape(lab_subtitle)}</div>
            </div>
        </div>
        <nav class="brand-links" aria-label="Project links">
{rendered_links}
        </nav>
    </div>
    <div class="brand-copy">
        <h1>{html.escape(product_name)}</h1>
        <p>{html.escape(description)}</p>
    </div>
    <div class="brand-pills" aria-label="Capabilities">
{rendered_capabilities}
    </div>
    <div id="build-badge">{runtime_html}</div>
</section>
"""
