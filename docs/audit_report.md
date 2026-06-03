# Corpus + labeling audit

_Generated 2026-05-22 15:17 UTC_

## 1. Corpus stats

| metric | value |
|---|---:|
| total_rows | 743,881 |
| submissions | 49,045 |
| comments | 693,221 |
| n_subreddits | 38 |
| n_unique_authors | 296,213 |
| date_first | 2013-05-01 |
| date_last | 2026-05-22 |
| avg_text_chars | 373.5 |
| median_text_chars | 221.0 |

## 2. Per-target positives by tier

| target | weak | disclosure | llm | manual | final_label |
|---|---:|---:|---:|---:|---:|
| anxiety | 68,937 (9.27%) | 390 (0.05%) | — | — | — |
| health_anxiety | 2,359 (0.32%) | 211 (0.03%) | — | — | — |
| depression | 10,094 (1.36%) | 931 (0.12%) | — | — | — |
| suicidality | 458 (0.06%) | 0 (0.00%) | — | — | — |

## 3. Top subreddits by total disclosures

| subreddit | n_posts | anxiety | health_anxiety | depression | suicidality | total |
|---|---:|---:|---:|---:|---:|---:|
| depression | 31,729 | 6 | 0 | 262 | 0 | **268** |
| BipolarReddit | 30,519 | 4 | 1 | 141 | 0 | **146** |
| Anxiety | 30,862 | 99 | 26 | 19 | 0 | **144** |
| HealthAnxiety | 19,098 | 18 | 86 | 9 | 0 | **113** |
| panicdisorder | 11,406 | 73 | 13 | 6 | 0 | **92** |
| mentalhealth | 27,974 | 11 | 1 | 77 | 0 | **89** |
| depression_help | 8,283 | 1 | 0 | 87 | 0 | **88** |
| AnxietyDepression | 7,626 | 17 | 5 | 49 | 0 | **71** |
| PanicAttack | 15,999 | 50 | 12 | 9 | 0 | **71** |
| COVID19_support | 21,121 | 12 | 18 | 20 | 0 | **50** |
| SuicideWatch | 23,629 | 1 | 0 | 49 | 0 | **50** |
| Anxietyhelp | 8,043 | 24 | 13 | 5 | 0 | **42** |
| BPD | 32,689 | 3 | 1 | 33 | 0 | **37** |
| agoraphobia | 12,730 | 16 | 6 | 7 | 0 | **29** |
| ChronicIllness | 21,051 | 1 | 6 | 17 | 0 | **24** |
| PTSD | 23,247 | 5 | 1 | 16 | 0 | **22** |
| CPTSD | 40,486 | 3 | 0 | 18 | 0 | **21** |
| ibs | 19,591 | 7 | 5 | 6 | 0 | **18** |
| AskDocs | 22,106 | 5 | 5 | 7 | 0 | **17** |
| OCD | 18,881 | 4 | 5 | 8 | 0 | **17** |
| dpdr | 7,353 | 1 | 0 | 16 | 0 | **17** |
| socialanxiety | 15,872 | 4 | 0 | 12 | 0 | **16** |
| relationship_advice | 54,477 | 1 | 0 | 12 | 0 | **13** |
| CasualConversation | 47,380 | 1 | 0 | 11 | 0 | **12** |
| CovidLongHaulers | 23,866 | 2 | 0 | 8 | 0 | **10** |

## 4. Disclosed users per target

| target | n disclosed users | avg posts/user | median | max |
|---|---:|---:|---:|---:|
| anxiety | 348 | 17.1 | 3.0 | 2722 |
| health_anxiety | 186 | 23.7 | 3.0 | 2722 |
| depression | 807 | 11.5 | 2.0 | 2722 |
| suicidality | 0 | None | None | None |

## 5. Example disclosure matches

These are random samples — useful for spotting regex false positives.

### anxiety

**1.** `r/panicdisorder` `id=i94y7dk`
  matched: `I have panic disorder`
  > It definitely feels very overwhelming at times especially when it's been a constant thing for a while. 
Do you have a lifeline support where you are ? Please do give them a call. It really does help to talk it out when you're feeling overwhelmed.
I have panic disorder and don't t

**2.** `r/Anxiety` `id=oj53zjc`
  matched: `I was diagnosed with GAD`
  > I have both and think my anxiety is worse. I functioned much better when I had depression alone before I was diagnosed with GAD.

**3.** `r/agoraphobia` `id=mvp1355`
  matched: `I have panic disorder`
  > There's no one true answer. I understand medication doesn't work for you, but if you can't function at all, you may need to see a different doctor who better understands this condition. I have panic disorder, GAD, acrophobia, and agoraphobia. I have managed these to a point where

**4.** `r/PTSD` `id=myms2m0`
  matched: `I have panic disorder`
  > I see where you're coming from. I have panic disorder and CPTSD and I've had people tell me my panic attacks are from my ptsd, when they are not (I'm diagnosed LMFAO). In total I've had two panic attacks from a ptsd episode, but I was triggered into them and I know why I was havi

**5.** `r/Anxietyhelp` `id=go7b6w6`
  matched: `I have GAD`
  > This is such good advice, thank you for taking the time to share, and for the book recommendations. I have GAD and PMDD and when the anxiety and panic really set in on the worst days, it feels like absolute hell. I refuse to let it keep taking over my life like this though and I 

### health_anxiety

**1.** `r/ChronicIllness` `id=1ozdpuj`
  matched: `I was diagnosed with health anxiety`
  > As a teenager I was diagnosed with "health anxiety disorder" by a child psychiatrist and have been systematically dismissed and neglected by every doctor I've seen since

When I was diagnosed with health anxiety disorder, my mother had just had a stroke at a relatively young age,

**2.** `r/BPD` `id=h14nyy1`
  matched: `I'm hypochondriac`
  > I shared it with my family...and they think I'm hypochondriac and that I diagnosed myself. My sisters are calling me spoiled brat, when I don't want to babysit their children, because I'm emotionally exhausted and that I don't want to screamcry in front of them. My parents won't 

**3.** `r/AnxietyDepression` `id=1o7g27l`
  matched: `I have health anxiety`
  > Has anyone had success with CBD for anxiety without feeling "different"?

I've had generalized anxiety disorder for about 5 years now. Been on SSRIs (Lexapro, then Zoloft), therapy, the whole routine. The meds help but they also make me feel... flat? Like my anxiety is lower but 

**4.** `r/PanicAttack` `id=1fh4n9j`
  matched: `i have health anxiety`
  > Panic attacks over body sensations 

is anyone just tired of this?? Whether it's heart palpitations, muscle cramps, migraines, some random numbness on face or body, or dissociation.. it just triggers my panic attacks since i have health anxiety and it's so bad! I try it calm myse

**5.** `r/Anxiety` `id=muoj8sq`
  matched: `I'm a hypochondriac`
  > But since I'm a hypochondriac one time isn't enough for me lol

### depression

**1.** `r/dpdr` `id=fia6p7v`
  matched: `I am depressed`
  > This was a really good response in terms of setting expectations. As I read the OP, I was thinking it might be disappointing to go through it all and have a totally normal result. All tests are normal (blood, etc) for me yet I am depressed. It's invisible and I often wish there w

**2.** `r/mentalhealth` `id=fkcb5q0`
  matched: `I have depression`
  > I have depression and anxiety I'm unhealthy over weight i have asthma and PCOS I'm 18 and in 6 days i have a flight to London from the USA (booked months ago before all this) I have no free cancelation and i don't have the money to just not go. I can't even sleep at night anymore

**3.** `r/SuicideWatch` `id=n9mimew`
  matched: `I'm depressed`
  > That actually makes sense (survival mode). I lost my job a while back and camping in state parks and felt invincible because all I could do was survive day by day. Now I have a job and I'm depressed. Ironic.

**4.** `r/depression` `id=1tddhm1`
  matched: `I'm depressed`
  > 18f lonely

I'm depressed and hypersomnic and so goddamn lonely. I just want a girlfriend so bad. Or just a girl to cuddle platonically. I don't know. Just so tired it's like moving through molasses to get through the day.

**5.** `r/AskDocs` `id=1ti0qgh`
  matched: `I'm depressed`
  > Is it possible that I've been misdiagnosed with depression and if so how can I go about getting proper treatment

I'm a 20 year old female, 5'5, 220 lbs. I've been taking zoloft 75mg for 4 years and buproprion for 8 months. I recently stopped taking buproprion because I found tha

### suicidality

_No positives in the corpus._

## 6. Notices

- ⚠️  suicidality: 0 disclosure positives — regex didn't match anything
