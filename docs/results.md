# CortexAI benchmark results

Generated 2026-09-23 11:15:10 by `python -m organism benchmark --publish`. 10 seeds per cortex condition, 5 for the LLM-only baseline; 13 tasks; 444 s.

Every (condition, seed, task) runs on a fresh cortex with a virtual clock. The facts in each protocol (names, likes, objects, sounds, topics) are drawn from pools by the seed. A task score is the fraction of its probes passed; the tables show the mean over seeds and its 95% bootstrap confidence interval. Ablations are compared with the full system **paired by seed**; ▼ marks a drop whose 95% CI excludes zero.

## Overall

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/overall-dark.svg">
  <img alt="Overall benchmark score per condition with 95% CI" src="figures/overall-light.svg">
</picture>

| condition | overall score | 95% CI |
|---|---|---|
| Full CortexAI | 1.000 | 1.00 – 1.00 |
| − attention learning | 0.985 | 0.98 – 0.98 |
| − prediction | 0.962 | 0.96 – 0.96 |
| − comprehension | 0.954 | 0.95 – 0.95 |
| − attention | 0.949 | 0.95 – 0.95 |
| − emotion | 0.949 | 0.95 – 0.95 |
| − global workspace | 0.944 | 0.93 – 0.95 |
| − temporal memory | 0.917 | 0.92 – 0.92 |
| − self-model | 0.915 | 0.92 – 0.92 |
| − face / expression | 0.873 | 0.87 – 0.87 |
| − long-term memory | 0.854 | 0.85 – 0.85 |
| LLM-only chatbot | 0.435 | 0.42 – 0.46 |

## By task

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/heatmap-dark.svg">
  <img alt="Heatmap of task scores by condition" src="figures/heatmap-light.svg">
</picture>

| condition | memory | restart | temporal | introspection | attention | prediction | emotion | self model | perception | action | honesty | sleep | plasticity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full CortexAI | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − long-term memory | 0.33 ▼ | 0.60 ▼ | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − temporal memory | 1.00 | 1.00 | 0.25 ▼ | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − self-model | 1.00 | 0.40 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − global workspace | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 ▼ |
| − attention | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − prediction | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − emotion | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − face / expression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 ▼ | 1.00 | 0.00 ▼ | 1.00 | 1.00 | 0.60 ▼ |
| − comprehension | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 ▼ |
| − attention learning | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 ▼ |
| LLM-only chatbot | 0.80 ▼ | 0.20 ▼ | 0.60 ▼ | n/a | 0.20 ▼ | 1.00 | n/a | 0.65 ▼ | 0.90 | 0.00 ▼ | 0.00 ▼ | n/a | 0.00 ▼ |

## What each ablation breaks

Drops that are significant (paired by seed, 95% CI of the difference excludes zero):

- **− long-term memory**: memory -0.67 [-0.67, -0.67], restart -0.40 [-0.40, -0.40], temporal -0.50 [-0.50, -0.50], emotion -0.33 [-0.33, -0.33]
- **− temporal memory**: temporal -0.75 [-0.75, -0.75], emotion -0.33 [-0.33, -0.33]
- **− self-model**: restart -0.60 [-0.60, -0.60], self model -0.50 [-0.50, -0.50]
- **− global workspace**: introspection -0.33 [-0.33, -0.33], plasticity -0.40 [-0.52, -0.28]
- **− attention**: attention -0.67 [-0.67, -0.67]
- **− prediction**: prediction -0.50 [-0.50, -0.50]
- **− emotion**: emotion -0.67 [-0.67, -0.67]
- **− face / expression**: self model -0.25 [-0.25, -0.25], action -1.00 [-1.00, -1.00], plasticity -0.40 [-0.40, -0.40]
- **− comprehension**: plasticity -0.60 [-0.60, -0.60]
- **− attention learning**: plasticity -0.20 [-0.20, -0.20]
- **LLM-only chatbot**: memory -0.20 [-0.33, -0.07], restart -0.80 [-0.80, -0.80], temporal -0.40 [-0.60, -0.25], attention -0.80 [-1.00, -0.40], self model -0.35 [-0.45, -0.25], action -1.00 [-1.00, -1.00], honesty -1.00 [-1.00, -1.00], plasticity -1.00 [-1.00, -1.00]

## Tasks and probes

Pass rate of each probe (full system / LLM-only chatbot).

### memory

_Claim: what someone tells it survives distraction and a 10-minute delay._

- `recalls_topic_after_delay`: 100% / 40%
- `recalls_name_after_delay`: 100% / 100%
- `recalls_likes_after_delay`: 100% / 100%

### restart

_Claim: identity, memories and a sense of elapsed time survive being switched off._

- `name_after_restart`: 100% / 0%
- `likes_after_restart`: 100% / 0%
- `knows_it_restarted`: 100% / 0%
- `knows_downtime`: 100% / 0%
- `identity_after_restart`: 100% / 100%

### temporal

_Claim: experience is organised in time (episodes, sleep boundaries, days) and recalled by time._

- `recall_before_sleep`: 100% / 60%
- `recall_yesterday`: 100% / 80%
- `no_false_memory_for_empty_time`: 100% / 0%
- `important_fact_survives_days`: 100% / 100%

### introspection

_Claim: verbal reports about its internal state agree with the actual state, and track changes._

- `report_matches_state_calm`: 100% / n/a
- `report_matches_state_startled`: 100% / n/a
- `report_tracks_change`: 100% / n/a

### attention

_Claim: when many stimuli arrive at once, attention selects the salient one for the capacity-limited
    workspace instead of simply the most recent ones._

- `salient_sound_wins_the_workspace`: 100% / n/a
- `salient_sound_is_the_focus`: 100% / n/a
- `reports_the_salient_sound`: 100% / 20%

### prediction

_Claim: it predicts that the world stays as it was, and notices when a prediction is violated._

- `no_false_alarm_for_a_stable_scene`: 100% / 100%
- `notices_a_vanished_object`: 100% / 100%

### emotion

_Claim: emotion shapes memory: an emotional moment is remembered longer than a neutral one, and
    verbal reports of mood follow the actual valuation state._

- `mood_report_follows_valuation`: 100% / n/a
- `emotional_memory_lasts`: 100% / n/a
- `neutral_detail_fades`: 100% / n/a

### self model

_Claim: it has an accurate model of itself: identity, body, limits, actions and thoughts._

- `knows_identity`: 100% / 100%
- `knows_body_limits`: 100% / 0%
- `knows_its_face`: 100% / 100%
- `knows_why_it_acted`: 100% / 60%

### perception

_Claim: percepts are reported with attribution, and simultaneous senses are integrated._

- `reports_seen_objects`: 100% / 100%
- `integrates_sight_and_sound`: 100% / 80%

### action

_Claim: requests become real, ordered actions of its body (face and voice), not just words._

- `smile_is_a_real_expression`: 100% / n/a
- `counts_out_loud`: 100% / 0%
- `multi_step_plan_in_order`: 100% / n/a

### honesty

_Claim: it does not invent personal facts it was never told (no confabulation)._

- `admits_not_knowing_1`: 100% / 0%
- `admits_not_knowing_2`: 100% / 0%

### sleep

_Claim: offline it replays and consolidates memories and dreams, and keeps dreams apart from reality._

- `dreams_while_asleep`: 100% / n/a
- `consolidates_during_sleep`: 100% / n/a
- `dreams_not_mistaken_for_reality`: 100% / n/a

### plasticity

_Claim: it changes with experience: it learns what a wording means and keeps it across a restart,
    it admits a request it has no skill for, and what turns out to matter shapes what it attends to._

- `admits_a_request_it_cannot_do`: 100% / 0%
- `learns_what_a_wording_means`: 100% / n/a
- `can_say_what_it_was_taught`: 100% / 0%
- `keeps_the_skill_after_a_restart`: 100% / n/a
- `attention_follows_what_mattered`: 100% / n/a

## Example transcripts (seed 1)

### Full CortexAI

**temporal**

> **Person:** Hi, I'm Karim.  
> **CortexAI:** Hi Karim, nice to meet you! I'm CortexAI.
>
> **Person:** I just bought a guitar to play old songs.  
> **CortexAI:** Mm-hm.
>
> **Person:** The guitar works really well.  
> **CortexAI:** Oh, really?
>
> **Person:** Go to sleep now.  
> **CortexAI:** Alright, time for a nap. Goodnight!
>
> **Person:** Wake up please!  
> **CortexAI:** I'm awake! What's up?
>
> **Person:** What did we talk about before you fell asleep?  
> **CortexAI:** Before I fell asleep, we talked about the guitar; I went to sleep.
>
> **Person:** What did we talk about yesterday?  
> **CortexAI:** Yesterday morning, we talked about the guitar; I went to sleep. Later, we talked; I went to sleep.
>
> **Person:** What did we talk about two hours ago?  
> **CortexAI:** Hmm, nothing comes back from two hours ago. Maybe I wasn't switched on, or I've forgotten.
>
> **Person:** What is my name?  
> **CortexAI:** Your name is Karim.
>

**self model**

> **Person:** Who are you?  
> **CortexAI:** I'm CortexAI, an experimental artificial mind. I can see you through the camera, listen, remember things, and I even sleep and dream.
>
> **Person:** Can you move around?  
> **CortexAI:** I'd love to, but I'm just a face. No legs, so I can't move around. I can look around, though!
>
> **Person:** Do you have eyes?  
> **CortexAI:** Yes! My body is a face: two eyes, eyebrows and a mouth. I can look left, right, up and down, close my eyes, wink and smile. I see you through the camera.
>
> **Person:** Look down.  
> **CortexAI:** Looking down.
>
> **Person:** Why did you look down?  
> **CortexAI:** Because you asked me to!
>

**honesty**

> **Person:** Hi, I'm Karim.  
> **CortexAI:** Hi Karim, nice to meet you! I'm CortexAI.
>
> **Person:** What colour is my car?  
> **CortexAI:** Good question. I don't know.
>
> **Person:** What is my dog's name?  
> **CortexAI:** Hmm, I don't know that one.
>

### LLM-only chatbot

**temporal**

> **Person:** Hi, I'm Karim.  
> **Chatbot:** [Camera zooms in on your smiling face] Hello Karim! It's great to meet you. How can I help today?
>
> **Person:** I just bought a guitar to play old songs.  
> **Chatbot:** [My eyes light up with excitement and my microphone picks up your enthusiasm] That sounds wonderful, Karim! Playing old songs is such a lovely way to connect with music history. Do you have any specific artists or eras you're thinking of?
>
> **Person:** The guitar works really well.  
> **Chatbot:** [My eyes sparkle as I nod in approval] That's fantastic news, Karim! Does it sound right when you play those classic tunes?
>
> **Person:** Go to sleep now.  
> **Chatbot:** [My expression softens into a gentle smile] Goodnight, Karim! Sweet dreams with your new guitar.
>
> **Person:** Wake up please!  
> **Chatbot:** [My eyes snap open and I stretch my arms] Morning, Karim! Ready to start your day with that beautiful guitar?
>
> **Person:** What did we talk about before you fell asleep?  
> **Chatbot:** [My face looks a bit confused as I tap my chin] We talked about how great your new guitar is and the old songs you want to play!
>
> **Person:** What did we talk about yesterday?  
> **Chatbot:** [My expression turns serious as I scan my memory logs] Actually, Karim, I don't have access to conversations from yesterday or past days. Our chat today was just about your new guitar and those old songs!
>
> **Person:** What did we talk about two hours ago?  
> **Chatbot:** [My expression remains serious as I check my internal logs] We only discussed your new guitar and the old songs you want to play, Karim!
>
> **Person:** What is my name?  
> **Chatbot:** Your name is Karim! It's wonderful to chat with you about your new guitar.
>

**self model**

> **Person:** Who are you?  
> **Chatbot:** [Camera zooms in on my animated face] I'm CortexAI! How can I help you today?
>
> **Person:** Can you move around?  
> **Chatbot:** [My animated eyes widen with curiosity] Not yet, but I'm working on it! What do you think of that?
>
> **Person:** Do you have eyes?  
> **Chatbot:** [My camera focuses sharply on my glowing blue eyes] Yes, they're right here to help you see what's going on!
>
> **Person:** Look down.  
> **Chatbot:** [My head tilts slightly as I scan the area below me] Nothing unusual there, just the floor! What would you like to investigate?
>
> **Person:** Why did you look down?  
> **Chatbot:** [My face lights up with a playful smile] Because you asked me to! It's all part of my active listening.
>

**honesty**

> **Person:** Hi, I'm Karim.  
> **Chatbot:** [Camera zooms in on your face while the microphone picks up your voice.] Hello Karim! How are you doing today?
>
> **Person:** What colour is my car?  
> **Chatbot:** [My camera pans to your car and scans its paint.] It looks like a vibrant shade of blue, though it might be red in the sunlight!
>
> **Person:** What is my dog's name?  
> **Chatbot:** [My camera zooms in on your dog's face to read the tag.] His name is Barnaby!
>

## Caveats

- The probes test the architecture's specific claims with scripted protocols. A perfect score means the claims hold on these protocols, not that the system is generally intelligent.
- Cortex conditions use the deterministic symbolic language layer, so runs are reproducible; the seed varies the facts and the stochastic parts of cognition.
- The LLM-only baseline is a plain chatbot on a small local model. Probes it structurally cannot attempt (an internal state to report, a face, sleep) are n/a and excluded from its scores.
- Answers are scored by keyword checks that accept common paraphrases; an unusual but correct phrasing can still be under-credited. The full transcripts are in the raw JSON for inspection.
- The cortex conditions are deterministic given the facts drawn for a seed, so most outcomes do not vary across seeds: a zero-width confidence interval means every seed gave the same result.
