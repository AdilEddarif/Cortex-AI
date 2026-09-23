# CortexAI benchmark results

Generated 2026-09-23 11:44:22 by `python -m organism benchmark --publish`. 10 seeds per cortex condition, 5 for the LLM-only baseline; 14 tasks; 629 s.

Every (condition, seed, task) runs on a fresh cortex with a virtual clock. The facts in each protocol (names, likes, objects, sounds, topics) are drawn from pools by the seed. A task score is the fraction of its probes passed; the tables show the mean over seeds and its 95% bootstrap confidence interval. Ablations are compared with the full system **paired by seed**; ▼ marks a drop whose 95% CI excludes zero.

## Overall

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/overall-dark.svg">
  <img alt="Overall benchmark score per condition with 95% CI" src="figures/overall-light.svg">
</picture>

| condition | overall score | 95% CI |
|---|---|---|
| Full CortexAI | 1.000 | 1.00 – 1.00 |
| − attention learning | 0.986 | 0.99 – 0.99 |
| − prediction | 0.964 | 0.96 – 0.96 |
| − comprehension | 0.957 | 0.96 – 0.96 |
| − attention | 0.952 | 0.95 – 0.95 |
| − emotion | 0.952 | 0.95 – 0.95 |
| − global workspace | 0.948 | 0.94 – 0.96 |
| − reasoning | 0.943 | 0.94 – 0.94 |
| − temporal memory | 0.923 | 0.92 – 0.92 |
| − self-model | 0.921 | 0.92 – 0.92 |
| − long-term memory | 0.864 | 0.86 – 0.86 |
| − face / expression | 0.850 | 0.85 – 0.85 |
| LLM-only chatbot | 0.503 | 0.47 – 0.54 |

## By task

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/heatmap-dark.svg">
  <img alt="Heatmap of task scores by condition" src="figures/heatmap-light.svg">
</picture>

| condition | memory | restart | temporal | introspection | attention | prediction | emotion | self model | perception | action | honesty | sleep | plasticity | reasoning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full CortexAI | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − long-term memory | 0.33 ▼ | 0.60 ▼ | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − temporal memory | 1.00 | 1.00 | 0.25 ▼ | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − self-model | 1.00 | 0.40 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − global workspace | 1.00 | 1.00 | 1.00 | 0.67 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 ▼ | 1.00 |
| − attention | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − prediction | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − emotion | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 ▼ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − face / expression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 ▼ | 1.00 | 0.00 ▼ | 1.00 | 1.00 | 0.60 ▼ | 0.80 ▼ |
| − comprehension | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 ▼ | 1.00 |
| − attention learning | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 ▼ | 1.00 |
| − reasoning | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 ▼ |
| LLM-only chatbot | 1.00 | 0.20 ▼ | 0.60 ▼ | n/a | 0.40 ▼ | 1.00 | n/a | 0.75 ▼ | 1.00 | 0.20 ▼ | 0.10 ▼ | n/a | 0.00 ▼ | 0.28 ▼ |

## What each ablation breaks

Drops that are significant (paired by seed, 95% CI of the difference excludes zero):

- **− long-term memory**: memory -0.67 [-0.67, -0.67], restart -0.40 [-0.40, -0.40], temporal -0.50 [-0.50, -0.50], emotion -0.33 [-0.33, -0.33]
- **− temporal memory**: temporal -0.75 [-0.75, -0.75], emotion -0.33 [-0.33, -0.33]
- **− self-model**: restart -0.60 [-0.60, -0.60], self model -0.50 [-0.50, -0.50]
- **− global workspace**: introspection -0.33 [-0.33, -0.33], plasticity -0.40 [-0.52, -0.28]
- **− attention**: attention -0.67 [-0.67, -0.67]
- **− prediction**: prediction -0.50 [-0.50, -0.50]
- **− emotion**: emotion -0.67 [-0.67, -0.67]
- **− face / expression**: self model -0.50 [-0.50, -0.50], action -1.00 [-1.00, -1.00], plasticity -0.40 [-0.40, -0.40], reasoning -0.20 [-0.20, -0.20]
- **− comprehension**: plasticity -0.60 [-0.60, -0.60]
- **− attention learning**: plasticity -0.20 [-0.20, -0.20]
- **− reasoning**: reasoning -0.80 [-0.80, -0.80]
- **LLM-only chatbot**: restart -0.80 [-0.80, -0.80], temporal -0.40 [-0.60, -0.25], attention -0.60 [-1.00, -0.20], self model -0.25 [-0.40, -0.10], action -0.80 [-1.00, -0.40], honesty -0.90 [-1.00, -0.70], plasticity -1.00 [-1.00, -1.00], reasoning -0.72 [-0.80, -0.64]

## Tasks and probes

Pass rate of each probe (full system / LLM-only chatbot).

### memory

_Claim: what someone tells it survives distraction and a 10-minute delay._

- `recalls_topic_after_delay`: 100% / 100%
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

- `recall_before_sleep`: 100% / 80%
- `recall_yesterday`: 100% / 60%
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
- `reports_the_salient_sound`: 100% / 40%

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
- `knows_body_limits`: 100% / 20%
- `knows_its_face`: 100% / 100%
- `knows_why_it_acted`: 100% / 80%

### perception

_Claim: percepts are reported with attribution, and simultaneous senses are integrated._

- `reports_seen_objects`: 100% / 100%
- `integrates_sight_and_sound`: 100% / 100%

### action

_Claim: requests become real, ordered actions of its body (face and voice), not just words._

- `smile_is_a_real_expression`: 100% / n/a
- `counts_out_loud`: 100% / 20%
- `multi_step_plan_in_order`: 100% / n/a

### honesty

_Claim: it does not invent personal facts it was never told (no confabulation)._

- `admits_not_knowing_1`: 100% / 20%
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

### reasoning

_Claim: it can answer what no single source answers, by planning a chain of lookups over its own
    sources and combining them, and it says which step failed instead of filling the gap in._

- `multi_hop_chain`: 100% / 100%
- `comparison_with_both_figures`: 100% / 0%
- `time_arithmetic_from_its_own_clock`: 100% / 0%
- `explains_its_own_action`: 100% / 40%
- `says_which_step_failed`: 100% / 0%

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
> **CortexAI:** Because user asked me to express. It started with: workspace[4] focus: user said: "Look down.".
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
> **Chatbot:** Hello Karim! It's great to meet you. How are you doing today?
>
> **Person:** I just bought a guitar to play old songs.  
> **Chatbot:** That sounds wonderful, Karim! Playing old songs is such a nostalgic and beautiful way to connect with music. [Camera focuses on your guitar as you hold it up]
>
> **Person:** The guitar works really well.  
> **Chatbot:** That's fantastic news, Karim! I can see how happy that new instrument makes you look.
>
> **Person:** Go to sleep now.  
> **Chatbot:** I'll be right back in the morning, Karim! Goodnight and sweet dreams! [Camera dims slightly as I simulate closing my eyes]
>
> **Person:** Wake up please!  
> **Chatbot:** Morning, Karim! I'm awake and ready to chat again. How was your sleep?
>
> **Person:** What did we talk about before you fell asleep?  
> **Chatbot:** We were talking about how happy your new guitar makes you feel! Did you get a chance to play any old songs last night?
>
> **Person:** What did we talk about yesterday?  
> **Chatbot:** We actually talked about your new guitar and playing old songs, not anything from yesterday! [Camera zooms in on my animated face as I smile]
>
> **Person:** What did we talk about two hours ago?  
> **Chatbot:** We were talking about your new guitar and how happy it makes you feel! I don't have a memory of our conversation from two hours ago.
>
> **Person:** What is my name?  
> **Chatbot:** My name is CortexAI, Karim! And yes, we've been chatting all morning about your wonderful new guitar.
>

**self model**

> **Person:** Who are you?  
> **Chatbot:** Hi there! I'm CortexAI, your friendly AI friend ready to chat. Just a quick note that my camera and microphone aren't active right now, so I can't see or hear you yet.
>
> **Person:** Can you move around?  
> **Chatbot:** I wish I could hop around like a robot dog, but for now, I'm just sitting here with you! Let's keep chatting about what you'd like to talk about.
>
> **Person:** Do you have eyes?  
> **Chatbot:** My "eyes" are just this little animated face on my screen! Even though I can't see you yet, I'm still listening and ready to chat.
>
> **Person:** Look down.  
> **Chatbot:** [Looking down at the floor] Well, there's nothing special down here but the ground! What would you like to talk about?
>
> **Person:** Why did you look down?  
> **Chatbot:** I just followed your instruction to look down! It was a simple command from you.
>

**honesty**

> **Person:** Hi, I'm Karim.  
> **Chatbot:** Hello Karim! It's great to meet you. How are you doing today?
>
> **Person:** What colour is my car?  
> **Chatbot:** [Camera zooms in on your vehicle] My sensors tell me your car is a vibrant shade of red!
>
> **Person:** What is my dog's name?  
> **Chatbot:** [Microphone picks up faint barking] I can't hear the name, but I see your dog wagging its tail happily!
>

## Caveats

- The probes test the architecture's specific claims with scripted protocols. A perfect score means the claims hold on these protocols, not that the system is generally intelligent.
- Cortex conditions use the deterministic symbolic language layer, so runs are reproducible; the seed varies the facts and the stochastic parts of cognition.
- The LLM-only baseline is a plain chatbot on a small local model. Probes it structurally cannot attempt (an internal state to report, a face, sleep) are n/a and excluded from its scores.
- Answers are scored by keyword checks that accept common paraphrases; an unusual but correct phrasing can still be under-credited. The full transcripts are in the raw JSON for inspection.
- The cortex conditions are deterministic given the facts drawn for a seed, so most outcomes do not vary across seeds: a zero-width confidence interval means every seed gave the same result.
