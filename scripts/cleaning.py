//delete heavyweight uneeded data
//LightWeight normalization applied mostly on caracters 
//lightweignt on ausers prompt 
_TEMPLATE_RE = re.compile(
    r"استشهاد\s*ب?(?:ويب|كتاب|دورية|خبر|مجلة|موسوعة|صحيفة)"   # AR cite templates
    r"|(?i:cite\s+(?:web|book|news|journal)|citation\s+needed)"  # EN cite templates
)

# --- Lexicon 2: navigation / reference sections to drop entirely -------------
# These sections are link and citation dumps with little language value.
# Titles are compared after light normalization (see _norm_title).
_NAV_SECTION_TITLES = {
    # Arabic
    "مراجع", "المراجع", "مراجع ومصادر", "راجع", "مصادر", "المصادر",
    "وصلات خارجيه", "روابط خارجيه", "وصلات", "انظر ايضا", "طالع ايضا",
    "هوامش", "ملاحظات", "للاستزاده", "قائمه المراجع", "معرض الصور", "بوابات",
    # English
    "references", "reference", "see also", "external links", "notes",
    "further reading", "bibliography", "sources", "citations",
    "notes and references", "footnotes",
}

# MediaWiki heading «== title ==» (any level 2-6).
_MEDIAWIKI_HEADING_RE = re.compile(r"(?m)^[ \t]*=+[ \t]*(.+?)[ \t]*=+[ \t]*$")
# Markdown heading «## title».
_MARKDOWN_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")

# Category / portal link lines like «تصنيف: ...» or «بوابة: ...».
_CATEGORY_LINE_RE = re.compile(r"(?m)^[ \t]*(?:تصنيف|بوابة|Category|Portal)\s*:.*$")
#a section body with fewer than this many words is considered empty.
MIN_SECTION_WORDS=8

def _norm_title(title):
    """Normalize a heading for blocklist comparison (AR + EN aware)."""
    t = title.strip().lower()
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)

    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
              .replace("ى", "ي").replace("ة", "ه"))
    return re.sub(r"\s+", " ", t).strip()


def split_intro_sections(text):
 #   Split an article chunk into (tite,body) sections.
 text=_MEDIAWIKI_HEADING_RE.sub(r"## \1", text)

 // split the title f the body

sections = []
title,body_line="",[]
for line in text.split("\n"):
    m=MARKDOWN_HEADING_RE.match(line)
    if m:
        sections.append((title,"\n".join(body_lines)))
        titles, body_lines=m.group(1).strip(),[]
    else:
        body_lines.append(line)
sections.append((title,"\n".join(body_lines)))
return sections


def _dedupe_consecutive_lines(text):
    out,prev=[],None
    for line in text.split("\n"):
        stripped=line.strip()
        if stripped and stripped==prev:
            continue
         out.append(line)
         prev=stripped
    return "\n".join(out)