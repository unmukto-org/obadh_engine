# Deliberate Input Rule Probe Corpus

This source-controlled corpus exercises rule signals, not memorized words. It is audit and regression material for the deterministic engine; it must not be loaded by the library, CLI, or WASM runtime.

| Category | Roman Input | Bengali Output | Contract |
|----------|-------------|----------------|----------|
| Vowel policy | `kOI kOU boi bou` | `কৈ কৌ বই বউ` | Uppercase `OI`/`OU` are diphthongs; lowercase `oi`/`ou` remain sequences. |
| Conjunct blocker | `kk kok kOk` | `ক্ক কক কোক` | Lowercase `o` blocks conjuncts without visible ও; uppercase `O` is visible. |
| Explicit hasant | `k,,k k,,a kk,,` | `ক্ক ক্আ ক্ক্` | `,,` is an explicit hasant/conjunct command, not a spelling guess. |
| Khanda-ta reph | <code>rrt rrt`` rr,,t`` rrt``sa</code> | `র্ত র্ৎ র্ৎ র্ৎসা` | `rrt` remains র্ত; খণ্ড ত uses the explicit <code>t``</code> signal. |
| Reph cluster | `rrkSh rrk,,Sh rrsk rrs,,ka` | `র্ক্ষ র্ক্ষ র্স্ক র্স্কা` | Reph composes over declared valid clusters, implicit or explicit. |
| Phola markers | `ky k,,w zya bwa Rw` | `ক্য ক্ব য্যা ব্বা ড়ওয়` | `y`/`w` are phola markers after a consonant base; `Rw` cannot form ড়্ব, so `w` renders as the standalone ওয় glide. |
| Productive ya-phola | `ply plYan blYak flYaT TrYak` | `প্ল্য প্ল্যান ব্ল্যাক ফ্ল্যাট ট্র্যাক` | ya-phola composes onto any conjunct base, including an r-phola tail (`TrYak` → ট্র্যাক for track), not only listed clusters. |
| Foreign letters | `q qq x X w iraq box water baqq` | `ক ঁ ক্স ক্স ওয় ইরাক বক্স ওয়াতের বাঁ` | Foreign-sound letters map by settled convention: `q`→ক, `x`→ক্স, `w`→ওয় glide; `qq` is the চন্দ্রবিন্দু signal, resolved ahead of `q` by longest match. |
| Base vs phola separation | `kz kb zoy zy bw` | `কয কব যয় য্য ব্ব` | `z`/`b` remain base letters; `y`/`w` become phola markers only through declared clusters. |
| Nasal signals | `ngg sMgo songskrriti shongkha` | `ঙ্গ সংগ সংস্কৃতি শংখা` | `ngg` is velar conjunct shorthand; `M` preserves literal anusvar. |
| Nasal escape | `songgo sMgo nggho sMgho` | `সঙ্গ সংগ ঙ্ঘ সংঘ` | `M` forces literal anusvar before `g`/`gh`; `ngg`/`nggh` remain deliberate velar conjunct shorthand. |
| Palatal nasal-ja | `NGj nj nJ jinjira noj nz` | `ঞ্জ ঞ্জ ঞ্জ জিঞ্জিরা নজ নয` | `NGj` is the source ঞ+জ signal; `nj`/`nJ` are narrow deterministic aliases, while `o` separates literal নজ and `nz` remains নয. |
| Jna pronunciation | `jNG jn gg ggan biggan gog` | `জ্ঞ জ্ঞ জ্ঞ জ্ঞান বিজ্ঞান গগ` | `gg` is a pronounced shorthand for orthographic জ্ঞ; use `gog` for literal গগ. |
| Long iya | `tiyw kiywo kiywO` | `তীয় কীয় কীয়ো` | `iyw` is the composable long-ঈয় signal. |
| Vocalic r | `rri krri rria rrhi` | `ঋ কৃ ঋআ র্হি` | `rri` is the vocalic ঋ signal; `rr` remains reph before consonants. |
| App vowel signal | `aYp AYp ayp Ayp app kaY kay` | `অ্যাপ অ্যাপ আয়প আয়প আপ্প ক্যা কায়` | `aY`/`AY` are অ্যা vowel signals; lowercase `y` remains the ordinary য় path. |
| Repeated vowel freedom | `a A aa i I ii e E ee o O oo u U uu kaa kee kii koo kuu kU uuupintocala` | `আ আ আআ ই ঈ ইই এ এ এএ অ ও অঅ উ ঊ উউ কাআ কেএ কিই কঅ কুউ কূ উউউপিন্তচালা` | Doubled lowercase vowels are not aliases; repeated signals remain repeated for deliberate invented strings. |
| Vowel sequences | `kai kau kia kio keo` | `কাই কাউ কিয়া কিও কেও` | Documented vowel sequences compose as rule units, not guessed spellings. |
| Marked vowel boundary | `k,,i k^a k:a` | `ক্ই কঁআ কঃআ` | Vowels after explicit hasant, chandrabindu, or bisarga start independently. |
| Non-conjunct ra-ya | `rZab rrYa Zya kZya` | `র‍্যাব র্যা য্যা কয্যা` | `rZ`/`rY` are the reph-ya markers; a stray `Z` folds to `z` (য). |
| Aspirated alias composition | `KhA KHy Chya jhya fya acCHHa` | `খা খ্য ছ্যা ঝ্যা ফ্যা আচ্ছা` | Accepted aspirated aliases canonicalize into ordinary rule components before vowel/conjunct handling. |
| Case fallback | `Biggan Ggan BhalO Khela Ga T D N Zya` | `বিজ্ঞান জ্ঞান ভালো খেলা গা ট ড ণ য্যা` | Unclaimed opposite-case rule signals fall back to the exact canonical signal; exact uppercase signals stay protected, and a stray `Z` folds to `z` (য). |
| External alias rejection | `Z pph p,,ph` | `য পফ প্ফ` | A stray `Z` folds to `z` (য); `Z` stays literal only inside the `rZ`/`rY` reph-ya marker, not as a blanket external-layout import. Explicit hasant stays available. Foreign letters with a settled convention (`q`→ক, `x`→ক্স, `w`→ওয়) are adopted for their own reason, see the Foreign letters row. |
| Symbols and numbers | `12.34 12.34. $` | `১২.৩৪ ১২.৩৪। ৳` | Decimal periods stay ASCII between number-bearing tokens. |
| Clock-time colon | `9:45 du:kho` | `৯:৪৫ দুঃখ` | `:` stays a literal separator between digits (clock times); between letters it is bisarga. |
