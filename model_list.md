# LeRobot ရှိ Models (Policies & Reward Models) စာရင်းနှင့် တွက်ချက်မှုဆိုင်ရာ ရှင်းလင်းချက်များ

LeRobot repository တွင် ပါဝင်သော Robotic Policies (စက်ရုပ်ထိန်းချုပ်မှု မော်ဒယ်များ)၊ Reward Models (ဆုကြေးနှင့် လုပ်ငန်းပြီးမြောက်မှု ခန့်မှန်း မော်ဒယ်များ) နှင့် RL Models အားလုံးကို အသေးစိတ် ခွဲခြမ်းစိတ်ဖြာ ဖော်ပြထားပါသည်။

---

## မာတိကာ (Table of Contents)

1. [အကျဉ်းချုပ် ခြုံငုံသုံးသပ်ချက် (Overview)](#၁-အကျဉ်းချုပ်-ခြုံငုံသုံးသပ်ချက်-overview)
2. [Robot Policy Models (စက်ရုပ်ထိန်းချုပ်မှု မော်ဒယ်များ - ၂၄ မျိုး)](#၂-robot-policy-models-စက်ရုပ်ထိန်းချုပ်မှု-မော်ဒယ်များ---၂၄-မျိုး)
   - [၁. ACT (Action Chunking with Transformers)](#၁-act-action-chunking-with-transformers)
   - [၂. Diffusion Policy](#၂-diffusion-policy)
   - [၃. DM05 (DM0.5)](#၃-dm05-dm05)
   - [၄. EO-1](#၄-eo-1)
   - [၅. Evo-1 (EVO1)](#၅-evo-1-evo1)
   - [၆. FastWAM](#၆-fastwam)
   - [၇. FineART-VLA](#၇-fineart-vla)
   - [၈. FLUX 3 Action (flux3)](#၈-flux-3-action-flux3)
   - [၉. Gaussian Actor](#၉-gaussian-actor)
   - [၁၀. GR00T N1.7](#၁၀-gr00t-n17)
   - [၁၁. LaWAM](#၁၁-lawam)
   - [၁၂. LingBot-VA](#၁၂-lingbot-va)
   - [၁၃. MolmoAct2](#၁၃-molmoact2)
   - [၁၄. Multi-Task DiT (multi_task_dit)](#၁၄-multi-task-dit-multi_task_dit)
   - [၁၅. π₀ (Pi0)](#၁၅-π₀-pi0)
   - [၁၆. π₀-FAST (Pi0-FAST)](#၁၆-π₀-fast-pi0-fast)
   - [၁၇. π₀.₅ (Pi0.5)](#၁၇-π₀₅-pi05)
   - [၁၈. RTC (Real-Time Chunking)](#၁၈-rtc-real-time-chunking)
   - [၁၉. SmolVLA](#၁၉-smolvla)
   - [၂၀. TD-MPC (Temporal Difference Model Predictive Control)](#၂၀-td-mpc-temporal-difference-model-predictive-control)
   - [၂၁. VLA-JEPA](#၂၁-vla-jepa)
   - [၂၂. VQ-BeT (Vector Quantized Behavior Transformer)](#၂၂-vq-bet-vector-quantized-behavior-transformer)
   - [၂၃. WALL-OSS / Wall-X](#၂၃-wall-oss--wall-x)
   - [၂၄. X-VLA](#၂၄-x-vla)
3. [Reward Models (ဆုကြေးနှင့် လုပ်ငန်းပြီးမြောက်မှု ခန့်မှန်း မော်ဒယ်များ - ၄ မျိုး)](#၃-reward-models-ဆုကြေးနှင့်-လုပ်ငန်းပြီးမြောက်မှု-ခန့်မှန်း-မော်ဒယ်များ---၄-မျိုး)
   - [၁. SARM (Stage-Aware Reward Model)](#၁-sarm-stage-aware-reward-model)
   - [၂. ROBOMETER](#၂-robometer)
   - [၃. TOPReward](#၃-topreward)
   - [၄. Reward Classifier](#၄-reward-classifier)
4. [Reinforcement Learning Algorithm (RL မော်ဒယ်)](#၄-reinforcement-learning-algorithm-rl-မော်ဒယ်)
   - [SAC (Soft Actor-Critic)](#sac-soft-actor-critic)
5. [အကျဉ်းချုပ် ဇယား (Comparison Summary Table)](#၅-အကျဉ်းချုပ်-ဇယား-comparison-summary-table)

---

## ၁. အကျဉ်းချုပ် ခြုံငုံသုံးသပ်ချက် (Overview)

LeRobot ရှိ မော်ဒယ်များကို အဓိကအားဖြင့် အောက်ပါအတိုင်း ခွဲခြားထားပါသည်-
- **VLA (Vision-Language-Action) Models**: ကင်မရာပုံရိပ်များ၊ သဘာဝဘာသာစကား ညွှန်ကြားချက်များ (Prompt) နှင့် စက်ရုပ်လက်ရှိ အနေအထား (Proprioception) တို့ကို ပေါင်းစပ်နားလည်ပြီး စက်ရုပ်လှုပ်ရှားမှု (Action Chunks) များကို ထုတ်ပေးသော Foundation Models ဖြစ်ကြပါသည်။
- **Imitation Learning & Diffusion Policies**: လူသားများ Teleoperation ဖြင့် မောင်းနှင်ထားသော ဒေတာများမှတစ်ဆင့် ဆက်တိုက်လှုပ်ရှားမှုများကို ချောမွေ့စွာ သင်ယူသော မော်ဒယ်များဖြစ်ပါသည်။
- **World Models & Video-Action Models**: အနာဂတ် ဗီဒီယိုပြကွက်များနှင့် လှုပ်ရှားမှုများကို ခန့်မှန်းတွက်ချက်ပေးသော မော်ဒယ်များဖြစ်ပါသည်။
- **Reward Models**: Reinforcement Learning (RL) သို့မဟုတ် Weighted Imitation Learning အတွက် စက်ရုပ်၏ လုပ်ငန်းပြီးမြောက်မှု အဆင့်နှင့် တိုးတက်မှုကို တွက်ချက်ပေးသော မော်ဒယ်များဖြစ်ပါသည်။

---

## ၂. Robot Policy Models (စက်ရုပ်ထိန်းချုပ်မှု မော်ဒယ်များ - ၂၄ မျိုး)

### ၁. ACT (Action Chunking with Transformers)
- **Identifier**: `act` (`ACTPolicy`, `ACTConfig`)
- **Architecture**: CVAE (Conditional Variational Autoencoder) + Transformer Encoder-Decoder + ResNet Backbone
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာပုံရိပ်များ (Multi-view RGB images) နှင့် စက်ရုပ် Joint Positions (Proprioception)
  - **Output**: အနာဂတ် လှုပ်ရှားမှုအဆင့်များပါဝင်သော Action Chunks (ဥပမာ- ရှေ့ ၁၀~၁၀၀ steps လှုပ်ရှားမှုအစီအစဉ်)
  - **တွက်ချက်ပုံ**: အချိန်တစ်ခုချင်းစီအတွက် Action တစ်ခုတည်းကို ခန့်မှန်းခြင်းမဟုတ်ဘဲ အနာဂတ် Action sequence အစုအပြုံ (Chunk) ကို တစ်ပြိုင်နက် ခန့်မှန်းတွက်ချက်ပါသည်။ Training ပြုလုပ်ချိန်တွင် CVAE ဖြင့် လူသားများ၏ မတူညီသော လှုပ်ရှားမှုပုံစံများကို Latent Space $z$ ထဲသို့ ဖမ်းယူပြီး၊ Inference အချိန်တွင် အမှားများ ဆင့်ပွားဖြစ်ပေါ်ခြင်း (Compounding Errors) ကို လျှော့ချကာ အလွန်တိကျချောမွေ့သော လှုပ်ရှားမှုများကို ဖြစ်ပေါ်စေပါသည်။
- **အသုံးပြုပုံ**: ALOHA ကဲ့သို့ လက်နှစ်ဖက်သုံး (Bimanual) စက်ရုပ်များတွင် အလွန်သေးငယ်တိကျသော ပစ္စည်းကိုင်တွယ်မှုများအတွက် သုံးပါသည်။
- **Reference**: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (2023)

---

### ၂. Diffusion Policy
- **Identifier**: `diffusion` (`DiffusionPolicy`, `DiffusionConfig`)
- **Architecture**: 1D Temporal CNN (ResNet-based) သို့မဟုတ် Transformer Backbone + DDPM/DDIM Noise Schedulers
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာပုံရိပ်များနှင့် စက်ရုပ်အနေအထား (State Observation)
  - **Output**: ချောမွေ့စွာ ထိန်းညှိပြီးသော စက်ရုပ် Action Trajectory
  - **တွက်ချက်ပုံ**: စက်ရုပ်လှုပ်ရှားမှု ထုတ်လုပ်ခြင်းကို Conditional Diffusion Process အဖြစ် ဖော်မြူလာဖွဲ့စည်းထားပါသည်။ ကျပန်းဆူညံသံ (Random Gaussian Noise) မှ စတင်ကာ Observation အခြေအနေပေါ်မူတည်၍ အဆင့်ဆင့် Noise များကို ဖယ်ထုတ် (Denoise) ခြင်းဖြင့် Multi-modal ဖြစ်သော လှုပ်ရှားမှုလမ်းကြောင်းများကို တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: ထိတွေ့ကိုင်တွယ်မှုများပြားသော (Contact-rich) လုပ်ငန်းများ၊ လူသားများပုံစံအမျိုးမျိုးဖြင့် ပြသထားသော သရုပ်ပြဒေတာများမှ လှုပ်ရှားမှုများကို သင်ယူရန် သုံးပါသည်။
- **Reference**: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023)

---

### ၃. DM05 (DM0.5)
- **Identifier**: `dm05` (`DM05Policy`, `DM05Config`)
- **Architecture**: Dexmal VLA Architecture + Flow Matching Action Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ဘာသာစကားညွှန်ကြားချက် (Language Instruction)၊ ကင်မရာပုံရိပ်များ (Front, Wrist) နှင့် Robot State
  - **Output**: စက်ရုပ် Continuous Action Chunks
  - **တွက်ချက်ပုံ**: Open-world Robot Control အတွက် ရည်ရွယ်ထုတ်လုပ်ထားသော VLA မော်ဒယ်ဖြစ်ပြီး၊ မြင်ကွင်းနှင့် ညွှန်ကြားချက်ကို အခြေခံ၍ Flow Matching နည်းပညာဖြင့် အနာဂတ် Action Chunks များကို လျင်မြန်စွာ တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: အသစ်အဆန်း ပတ်ဝန်းကျင်များတွင် စက်ရုပ်များ အထွေထွေ အသုံးပြုနိုင်ရန် (ဥပမာ- LIBERO Benchmark တွင် စံချိန်တင် အောင်မြင်မှုရရှိထားပါသည်)။
- **Reference**: Dexmal Team, *DM0.5: An Open-World Foundation Model for General-Purpose Embodied Intelligence* (2026)

---

### ၄. EO-1
- **Identifier**: `eo1` (`EO1Policy`, `EO1Config`)
- **Architecture**: Qwen2.5-VL Backbone + Continuous Flow-Matching Action Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Multimodal Dialogue (ရုပ်ပုံများ၊ စကားလုံးညွှန်ကြားချက်များနှင့် State Tokens)
  - **Output**: EO-1 Action Tokens မှတစ်ဆင့် စက်ရုပ် Continuous Actions
  - **တွက်ချက်ပုံ**: စက်ရုပ်ထိန်းချုပ်မှုကို Multimodal Conversation တစ်ခုသဖွယ် ပုံဖော်ထားပါသည်။ Qwen2.5-VL ဖြင့် ရုပ်ပုံနှင့် ဘာသာစကားကို ပေါင်းစပ်နားလည်ပြီး Flow-matching Head မှ Action Chunks များကို Denoise လုပ်ကာ တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: စကားဖြင့် ညွှန်ကြားချက်ပေးကာ စက်ရုပ်ကို အမျိုးစုံ ထိန်းချုပ်စေခိုင်းရန် သုံးပါသည်။
- **Reference**: Song et al., *EO-1 Vision-Language-Action Policy* (2025)

---

### ၅. Evo-1 (EVO1)
- **Identifier**: `evo1` (`Evo1Policy`, `Evo1Config`)
- **Architecture**: InternVL3 Vision-Language Backbone + Flow-Matching Action Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ဘာသာစကား ညွှန်ကြားချက်၊ Multi-view Images
  - **Output**: စက်ရုပ် Action Chunks
  - **တွက်ချက်ပုံ**: InternVL3 Backbone မှ ထုတ်ပေးသော Multimodal Embeddings များကို အခြေခံ၍ Flow Matching Action Head ဖြင့် စဉ်ဆက်မပြတ် စက်ရုပ် Action Trajectories များကို တွက်ထုတ်ပေးပါသည်။ Staged training နှင့် Real-time Chunking (RTC) တို့ကို ထောက်ပံ့ပေးပါသည်။
- **အသုံးပြုပုံ**: တရုတ် Shanghai Jiao Tong University (MINT-SJTU) မှ ထုတ်လုပ်ထားပြီး ရှုပ်ထွေးသော စက်ရုပ်ထိန်းချုပ်မှုများအတွက် အသုံးပြုပါသည်။
- **Reference**: MINT-SJTU, *Evo-1 Policy* (2025)

---

### ၆. FastWAM
- **Identifier**: `fastwam` (`FastWAMPolicy`, `FastWAMConfig`)
- **Architecture**: Wan2.2-TI2V-5B Video Diffusion Stack + Fast Direct Action Predictor
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Video Frames၊ Language Goal၊ Robot State
  - **Output**: စက်ရုပ် Action Chunks `[batch, action_horizon, action_dim]`
  - **တွက်ချက်ပုံ**: Training ပြုလုပ်ချိန်တွင် ဗီဒီယို Visual World Dynamics များကို သင်ယူထားသော်လည်း၊ Inference (စက်ရုပ်မောင်းနှင်ချိန်) တွင် ဗီဒီယိုအသစ်များကို အဆင့်ဆင့်ထုတ်လုပ်မနေဘဲ Direct Action Prediction ကို ပြုလုပ်ပေးသောကြောင့် အလွန်မြန်ဆန်စွာ (High Frequency) တွက်ထုတ်ပေးနိုင်ပါသည်။
- **အသုံးပြုပုံ**: ကမ္ဘာကြီး၏ ရူပဗေဒသဘောတရားကို နားလည်သော World Model ကို အခြေခံပြီး တုံ့ပြန်မှု မြန်ဆန်သော စက်ရုပ်ထိန်းချုပ်မှု လိုအပ်သည့်အခါ သုံးပါသည်။
- **Reference**: Yuan et al., *FastWAM: World Action Model for Robot Control* (2026)

---

### ၇. FineART-VLA
- **Identifier**: `fineart_vla` (`FineARTVLAPolicy`, `FineARTVLAConfig`)
- **Architecture**: π₀.₅ + Dual-System (System 2 Language Head + System 1 Flow-Matching Action Expert)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ရုပ်ပုံများ၊ စက်ရုပ်အနေအထားနှင့် လုပ်ဆောင်ရမည့် ပင်မရည်မှန်းချက်
  - **Output**: ရှေ့ဆက်လုပ်ရမည့် Subtask (သဘာဝဘာသာစကားဖြင့်) နှင့် တိုက်ရိုက်ထိန်းချုပ်မည့် Action Chunks
  - **တွက်ချက်ပုံ**: လူသားတို့၏ ဦးနှောက်သဖွယ် System 2 ဖြင့် ဘာသာစကားအရ တွေးခေါ်ဆုံးဖြတ်ပြီး (Subtask prediction)၊ ထိုဆုံးဖြတ်ချက်ကို System 1 Action Expert ထံ ပေးပို့ကာ အချိန်နှင့်တပြေးညီ Flow Matching ဖြင့် စက်ရုပ်လက်မောင်းကို မောင်းနှင်စေပါသည်။ Knowledge Insulation စနစ်ပါရှိသောကြောင့် Action Gradients များက Pretrained VLM စွမ်းရည်ကို ပျက်စီးစေခြင်း မရှိပါ။
- **အသုံးပြုပုံ**: အဆင့်ပေါင်းများစွာ ပါဝင်သော ရှည်လျားသည့် လက်နှစ်ဖက်သုံး အလုပ်များကို စနစ်တကျ ခွဲခြမ်းလုပ်ဆောင်ရန် သုံးပါသည်။ (Scale AI & Hugging Face)
- **Reference**: Scale AI & Hugging Face, *FineART-VLA* (2026)

---

### ၈. FLUX 3 Action (flux3)
- **Identifier**: `flux3` (`Flux3Policy`, `Flux3Config`)
- **Architecture**: Black Forest Labs FLUX.3 Diffusion Transformer Trunk + Action Head + Frozen Video VAE
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Camera Canvas (ရုပ်ပုံများ)၊ State History နှင့် Task Instruction
  - **Output**: Denoised Robot Action Chunks
  - **တွက်ချက်ပုံ**: Video Latents နှင့် Action Chunks များကို FLUX.3 Diffusion Transformer ပေါ်တွင် ပူးတွဲ Denoise လုပ်ကာ တွက်ထုတ်ပါသည်။ စက်ရုပ်အမျိုးအစားအသစ်များအတွက် LoRA (PEFT) ဖြင့် Trunk ကို ထိန်းညှိပြီး Action Head ကို အပြည့်အဝ Train နိုင်ပါသည်။
- **အသုံးပြုပုံ**: SO-101 နှင့် Franka Panda စက်ရုပ်များတွင် အစွမ်းထက် visual features များကို အသုံးပြု၍ Action မောင်းနှင်ရန် သုံးပါသည်။
- **Reference**: Black Forest Labs & LeRobot, *FLUX 3 Action Policy* (2026)

---

### ၉. Gaussian Actor
- **Identifier**: `gaussian_actor` (`GaussianActorPolicy`, `GaussianActorConfig`)
- **Architecture**: Visual Encoder (ResNet/CNN) + MLP Gaussian Head (Tanh Squashed) + Critic Networks
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ရုပ်ပုံနှင့် စက်ရုပ် လက်ရှိ အနေအထား
  - **Output**: Action Probability Distribution ၏ ပျမ်းမျှတန်ဖိုး ($\mu$) နှင့် စံသွေဖည်ကိန်း ($\sigma$) -> Continuous Action Sample
  - **တွက်ချက်ပုံ**: Reinforcement Learning (အထူးသဖြင့် Soft Actor-Critic / SAC) algorithm များအတွက် စက်ရုပ်၏ Continuous Action များကို Gaussian Distribution မှတစ်ဆင့် Sample ထုတ်ယူ တွက်ချက်ပေးပါသည်။ Exploration (စူးစမ်းရှာဖွေခြင်း) နှင့် Exploitation ကို မျှတစေပါသည်။
- **အသုံးပြုပုံ**: HIL-SERL (Human-in-the-Loop Reinforcement Learning) နှင့် Online/Offline စက်ရုပ် RL လေ့ကျင့်မှုများတွင် သုံးပါသည်။
- **Reference**: Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* (2018)

---

### ၁၀. GR00T N1.7
- **Identifier**: `groot` (`GrootPolicy`, `GrootConfig`)
- **Architecture**: NVIDIA Cosmos-Reason2 / Qwen3-VL Backbone + Flow-Matching Action Transformer
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Language Instructions၊ Multi-view Images၊ Proprioception
  - **Output**: စက်ရုပ် Action Chunks
  - **တွက်ချက်ပုံ**: NVIDIA ၏ လူတူစက်ရုပ် (Humanoid) နှင့် Cross-embodiment Foundation Model ဖြစ်ပြီး၊ မြောက်မြားစွာသော တကယ့်စက်ရုပ်ဒေတာများနှင့် Isaac Sim Synthetic ဒေတာများဖြင့် Train ထားပါသည်။ Observation များမှတစ်ဆင့် Flow Matching Action Transformer ဖြင့် လှုပ်ရှားမှုများကို ချောမွေ့စွာ တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: လူတူစက်ရုပ်များ (Humanoids) နှင့် ရှုပ်ထွေးသော စက်ရုပ်လက်မောင်းများစွာတွင် တစ်ပြိုင်နက် သုံးနိုင်သော Generalist မော်ဒယ်အဖြစ် သုံးပါသည်။
- **Reference**: NVIDIA, *GR00T N1.7: Generalist Humanoid Robot Reasoning and Skills* (2025)

---

### ၁၁. LaWAM
- **Identifier**: `lawam` (`LaWAMPolicy`, `LaWAMConfig`)
- **Architecture**: Qwen3-VL Backbone + DINOv3 Latent Action/World Model (LAM) + Flow-Matching Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: လက်ရှိ ပုံရိပ်များ၊ ဘာသာစကား ရည်မှန်းချက်၊ စက်ရုပ် အနေအထား
  - **Output**: Latent World Dynamics အခြေအနေနှင့် စက်ရုပ် Action Chunks
  - **တွက်ချက်ပုံ**: Latent Space ထဲတွင် ကမ္ဘာ့ရူပဗေဒအခြေအနေ အပြောင်းအလဲ (Dynamics) များကို ဦးစွာ ကြိုတင်တွက်ချက်ပြီး၊ ထို Dynamics အပေါ် အခြေခံကာ Flow Matching ဖြင့် အကောင်းဆုံး Action များကို ထုတ်လုပ်ပေးပါသည်။
- **အသုံးပြုပုံ**: ရူပဗေဒအရ ပစ္စည်းများ ရွေ့လျားမှုနှင့် ထိခိုက်မှု အကျိုးဆက်များကို နားလည်စေလိုသော အဆင့်မြင့် စက်ရုပ်လုပ်ငန်းများတွင် သုံးပါသည်။
- **Reference**: RLinf, *LaWAM: Latent World Action Model for Dynamics-Aware Robot Policies* (2026)

---

### ၁၂. LingBot-VA
- **Identifier**: `lingbot_va` (`LingBotVAPolicy`, `LingBotVAConfig`)
- **Architecture**: Wan2.2 Video-Diffusion Stack (Autoregressive Dual-Stream Mixture-of-Transformers)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Video Frames၊ History Keyframes၊ Language Instructions
  - **Output**: အနာဂတ် ဗီဒီယို Latent များ (Future Video Latents) နှင့် စက်ရုပ် Actions များကို တစ်ပြိုင်နက် ရောနှောထုတ်ပေးခြင်း
  - **တွက်ချက်ပုံ**: ဗီဒီယိုခန့်မှန်းခြင်းနှင့် စက်ရုပ်လှုပ်ရှားမှုတို့ကို အစဉ်လိုက် တတွဲတည်း (Interleaved Autoregressive Sequence) အဖြစ် တွက်ထုတ်ပေးပါသည်။ စက်ရုပ်သည် "ငါ ဒီလိုလှုပ်ရှားလိုက်ရင် မြင်ကွင်းက ဘာဖြစ်သွားမလဲ" ဟူသော အနာဂတ်မြင်ကွင်းကို စိတ်ကူးပုံဖော်ရင်း Action များကို တွက်ချက်ပါသည်။
- **အသုံးပြုပုံ**: Video-Action ပူးတွဲလေ့လာမှုများနှင့် Long-horizon စက်ရုပ်စီမံခန့်ခွဲမှုများတွင် သုံးပါသည်။
- **Reference**: Robbyant, *LingBot-VA: Autoregressive Video-Action World Model* (2025)

---

### ၁၃. MolmoAct2
- **Identifier**: `molmoact2` (`MolmoAct2Policy`, `MolmoAct2Config`)
- **Architecture**: Allen AI Molmo Vision-Language Backbone + Per-layer Flow-Matching Action Expert
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Multi-camera Images နှင့် Natural Language Prompts
  - **Output**: စက်ရုပ် Continuous Action Chunks
  - **တွက်ချက်ပုံ**: Molmo Vision-Language Model ၏ Layer တစ်ခုချင်းစီတွင် Flow-Matching Action Expert များကို ချိတ်ဆက်ထားပြီး၊ အမြင်အာရုံနှင့် စကားလုံးအဓိပ္ပာယ်ကို နက်နက်ရှိုင်းရှိုင်း ခွဲခြမ်းစိတ်ဖြာကာ တိကျသော စက်ရုပ် Action Trajectories များကို တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: Allen AI ၏ ပွင့်လင်းမြင်သာပြီး အဆင့်မြင့် Foundation Model စွမ်းဆောင်ရည်ကို စက်ရုပ်နယ်ပယ်တွင် အသုံးချရန် ဖြစ်ပါသည်။
- **Reference**: Allen Institute for AI (Ai2), *MolmoAct2* (2025)

---

### ၁၄. Multi-Task DiT (multi_task_dit)
- **Identifier**: `multi_task_dit` (`MultiTaskDiTPolicy`, `MultiTaskDiTConfig`)
- **Architecture**: CLIP ViT-B/16 Vision/Text Encoders + Large Diffusion Transformer (DiT ~450M params)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ဘာသာစကား ညွှန်ကြားချက်၊ Multi-camera Images၊ စက်ရုပ် State
  - **Output**: Action Trajectory (Diffusion သို့မဟုတ် Flow-matching Objective ဖြင့်)
  - **တွက်ချက်ပုံ**: ကွန်ပျူတာဗျူဟာကြီးမားသော Diffusion Transformer ကို အသုံးပြုပြီး စာသားနှင့် ပုံရိပ် Feature များကို Conditioning အဖြစ် ပေးသွင်းကာ လုပ်ငန်းပေါင်းစုံ (Multi-task) အတွက် Action များကို Generative Denoising ဖြင့် တွက်ထုတ်ပေးပါသည်။ `torch.compile` ဖြင့် အလွန်မြန်ဆန်စွာ run နိုင်ပါသည်။
- **အသုံးပြုပုံ**: လုပ်ငန်းစဉ်ပေါင်းများစွာကို မော်ဒယ်တစ်ခုတည်းဖြင့် ကျွမ်းကျင်စွာ သင်ယူစေလိုသည့်အခါ သုံးပါသည်။
- **Reference**: LeRobot Multi-Task DiT Policy (2025)

---

### ၁၅. π₀ (Pi0)
- **Identifier**: `pi0` (`PI0Policy`, `PI0Config`)
- **Architecture**: PaliGemma Vision-Language Backbone + Flow-Matching Action Expert Transformer
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာမြင်ကွင်းများ၊ ဘာသာစကား ညွှန်ကြားချက်၊ စက်ရုပ် လက်ရှိအနေအထား
  - **Output**: High-frequency စက်ရုပ် Action Chunks (50 Hz အထိ)
  - **တွက်ချက်ပုံ**: Physical Intelligence ၏ နာမည်ကျော် ပထမဆုံး Generalist Robot Foundation Model ဖြစ်ပါသည်။ ပုံရိပ်နှင့် ဘာသာစကားကို PaliGemma ဖြင့် နားလည်စေပြီး သီးသန့် Action Expert Transformer မှတစ်ဆင့် Flow Matching ဖြင့် စဉ်ဆက်မပြတ် စက်ရုပ် Action များကို မြန်ဆန်တိကျစွာ တွက်ချက်ထုတ်လုပ်ပေးပါသည်။
- **အသုံးပြုပုံ**: စားပွဲရှင်းခြင်း၊ အဝတ်ခေါက်ခြင်း၊ ပစ္စည်းထည့်ခြင်းစသည့် အထွေထွေ အိမ်တွင်း/စက်ရုံ လုပ်ငန်းများအတွက် သုံးပါသည်။
- **Reference**: Physical Intelligence, *π₀: A Vision-Language-Action Flow Model for General Robot Control* (2024)

---

### ၁၆. π₀-FAST (Pi0-FAST)
- **Identifier**: `pi0_fast` (`PI0FastPolicy`, `PI0FastConfig`)
- **Architecture**: Gemma Vision-Language Backbone + FAST (Frequency-space Action Sequence Tokenization)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာပုံရိပ်များနှင့် စကားလုံးညွှန်ကြားချက်
  - **Output**: Discrete Action Tokens (Autoregressive Next-Token Prediction ဖြင့်)
  - **တွက်ချက်ပုံ**: စက်ရုပ် Action များကို Diffusion/Flow-matching ဖြင့် မတွက်ဘဲ၊ FAST နည်းပညာဖြင့် Frequency Space ထဲတွင် Tokenize လုပ်ကာ ဘာသာစကား Model များအတိုင်း Next-token Prediction (Autoregressive) ဖြင့် တွက်ထုတ်ပါသည်။ ထို့ကြောင့် Diffusion မော်ဒယ်များထက် လေ့ကျင့်မှု ၅ ဆ ပိုမိုမြန်ဆန်ပါသည်။
- **အသုံးပြုပုံ**: အလွန်လျင်မြန်သော လေ့ကျင့်မှု (Fast Training) လိုအပ်ပြီး တိကျကျွမ်းကျင်သော လှုပ်ရှားမှုများ လုပ်ဆောင်ရန် သုံးပါသည်။
- **Reference**: Physical Intelligence, *π₀-FAST: Autoregressive Action Tokenization for Dexterous Robotics* (2024)

---

### ၁၇. π₀.₅ (Pi0.5)
- **Identifier**: `pi05` (`PI05Policy`, `PI05Config`)
- **Architecture**: အဆင့်မြှင့်တင်ထားသော VLM Backbone + Flow Matching Action Head + RTC Module
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာရုပ်ပုံများ၊ ပတ်ဝန်းကျင်အသစ်များမှ မြင်ကွင်းများ၊ စကားလုံးညွှန်ကြားချက်
  - **Output**: Open-world Generalized Action Chunks
  - **တွက်ချက်ပုံ**: π₀ ကို Open-world Generalization ရရှိစေရန် အဆင့်မြှင့်ထားခြင်းဖြစ်ပြီး၊ တစ်ခါမှမမြင်ဖူးသော ပတ်ဝန်းကျင်အသစ်များနှင့် မတူညီသော ပစ္စည်းကိရိယာများကို ရူပဗေဒနှင့် အဓိပ္ပာယ်သဘောတရားအရ နားလည်နိုင်စွမ်းရှိစေရန် တွက်ချက်ပေးပါသည်။ Real-Time Chunking စနစ်ပါဝင်ပြီး တုံ့ဆိုင်းမှုမရှိ ချောမွေ့စွာ လှုပ်ရှားနိုင်ပါသည်။
- **အသုံးပြုပုံ**: မရင်းနှီးသော အိမ်များနှင့် ပတ်ဝန်းကျင်အသစ်များတွင် စက်ရုပ်ကို ချက်ချင်းအသုံးချနိုင်စေရန် ရည်ရွယ်ပါသည်။
- **Reference**: Physical Intelligence, *π₀.₅: Open-World Generalization in Robotics* (2025)

---

### ၁၈. RTC (Real-Time Chunking)
- **Identifier**: `rtc` (`RTCConfig`, `ActionQueue`, `ActionInterpolator`)
- **Architecture**: Inference Runtime Execution Engine (Queue, Latency Compensation, Action Prefix Anchoring)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Model Inference Latency၊ စက်ရုပ်လက်ရှိ အချိန်နှင့် အနေအထား၊ မော်ဒယ်မှ ထွက်လာသော Action Chunks
  - **Output**: အချိန်နှင့်တပြေးညီ တုံ့ဆိုင်းမှုမရှိ ဆက်တိုက်ပို့လွှတ်ရမည့် Interpolated Robot Control Commands
  - **တွက်ချက်ပုံ**: မော်ဒယ်တစ်ခုတည်း မဟုတ်ဘဲ Flow-matching အခြေခံ VLA မော်ဒယ်များ (π₀, π₀.₅, SmolVLA စသည်) အတွက် Runtime Engine ဖြစ်ပါသည်။ မော်ဒယ်ကြီးများ စဉ်းစားတွက်ချက်နေချိန် (Inference Latency) အတွင်း စက်ရုပ် ရပ်တန့်မသွားစေရန် လက်ရှိရွေ့လျားနေဆဲ Action Chunk နှင့် နောက်ထပ်ရောက်လာမည့် Action Chunk တို့ကို ချောမွေ့စွာ ဆက်စပ်တွက်ချက်ပေးပါသည်။
- **အသုံးပြုပုံ**: VLA မော်ဒယ်ကြီးများကို စက်ရုပ်အစစ်များတွင် တုံ့ဆိုင်းရပ်တန့်မှုမရှိဘဲ ချောမွေ့စွာ ခုတ်မောင်းရန် သုံးပါသည်။
- **Reference**: Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow Policies* (2025)

---

### ၁၉. SmolVLA
- **Identifier**: `smolvla` (`SmolVLAPolicy`, `SmolVLAConfig`)
- **Architecture**: Hugging Face SmolLM/Vision Backbone + Flow-Matching Action Expert (Lightweight Architecture)
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Multi-camera Images၊ Language Instructions၊ Joint Positions
  - **Output**: စက်ရုပ် Action Chunks
  - **တွက်ချက်ပုံ**: ကွန်ပျူတာ အရင်းအမြစ် နည်းပါးသော သာမန် Consumer GPU (ဥပမာ RTX 3090/4090) များပေါ်တွင်ပါ အလွယ်တကူ Fine-tune လုပ်နိုင်ပြီး Memory အသုံးပြုမှု အလွန်နည်းပါးစေရန် Flow Matching နည်းပညာဖြင့် လှုပ်ရှားမှုများကို အကျဉ်းချုံး တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: သုတေသီများနှင့် ဝါသနာရှင်များအတွက် စရိတ်သက်သာစွာဖြင့် VLA စက်ရုပ်မော်ဒယ်များကို စမ်းသပ်မောင်းနှင်ရန် အထူးသင့်လျော်ပါသည်။
- **Reference**: Hugging Face, *SmolVLA: Compact Vision-Language-Action Model* (2025)

---

### ၂၀. TD-MPC (Temporal Difference Model Predictive Control)
- **Identifier**: `tdmpc` (`TDMPCPolicy`, `TDMPCConfig`)
- **Architecture**: Latent World Model (Encoder, Dynamics, Reward Head, Value Head) + MPPI/CEM Optimizer
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: လက်ရှိ အမြင်အာရုံ သို့မဟုတ် State Observation
  - **Output**: Latent Space ထဲတွင် Trajectory Optimization ပြုလုပ်ပြီး ရရှိလာသော အကောင်းဆုံး Action
  - **တွက်ချက်ပုံ**: ပတ်ဝန်းကျင်၏ Latent Representation ကို သင်ယူပြီး၊ Temporal Difference (TD) Learning ဖြင့် Q-value နှင့် Reward များကို ခန့်မှန်းပါသည်။ Inference အချိန်တွင် MPPI (Model Predictive Path Integral) သို့မဟုတ် CEM ဖြင့် Latent အနာဂတ်လမ်းကြောင်း ထောင်ပေါင်းများစွာကို လျင်မြန်စွာ စမ်းသပ်ရှာဖွေပြီး ဆုကြေး (Return) အများဆုံးရမည့် ပထမဆုံး Action ကို ရွေးချယ်တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: Offline သို့မဟုတ် Online ဒေတာများမှတစ်ဆင့် စက်ရုပ်ကို Model-based RL ဖြင့် စွမ်းဆောင်ရည်မြင့်မားစွာ ထိန်းချုပ်ရန် သုံးပါသည်။
- **Reference**: Hansen, Wang & Su, *Temporal Difference Learning for Model Predictive Control* (ICML 2022)

---

### ၂၁. VLA-JEPA
- **Identifier**: `vla_jepa` (`VLAJEPAPolicy`, `VLAJEPAConfig`)
- **Architecture**: Qwen3-VL Language Backbone + V-JEPA2 (Video World Model) + DiT Flow-Matching Action Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Multi-view Video Observation နှင့် Language Instruction
  - **Output**: Flow-matching စက်ရုပ် Action Chunks
  - **တွက်ချက်ပုံ**: ရုပ်ပုံများကို pixel အဆင့် generate လုပ်မနေဘဲ၊ Meta ၏ Non-generative V-JEPA2 ဖြင့် အဆင့်မြင့် Spatio-temporal visual embeddings များကို ထုတ်ယူပါသည်။ ၎င်းကို Qwen3-VL ဘာသာစကား Context နှင့် ပေါင်းစပ်ကာ Diffusion Transformer (DiT) Action Head ဖြင့် အနာဂတ် Action များကို တိကျစွာ ခန့်မှန်းတွက်ချက်ပေးပါသည်။
- **အသုံးပြုပုံ**: ရှုပ်ထွေးသော visual dynamics ရှိသည့် အခြေအနေများတွင် စွမ်းဆောင်ရည်မြင့် robot manipulation အတွက် သုံးပါသည်။
- **Reference**: starVLA / LeRobot, *VLA-JEPA: Non-generative World Model Representations for Robotics* (2026)

---

### ၂၂. VQ-BeT (Vector Quantized Behavior Transformer)
- **Identifier**: `vqbet` (`VQBeTPolicy`, `VQBeTConfig`)
- **Architecture**: VQ-VAE (Vector Quantized Action Codebook) + minGPT-style Transformer Backbone + Offset Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: History of Visual Observations and States
  - **Output**: Discrete Cluster + Continuous Residual Offset ပေါင်းစပ်ထားသော Action
  - **တွက်ချက်ပုံ**: စက်ရုပ်၏ Continuous Action များကို VQ-VAE ဖြင့် ကိန်းဂဏန်း အုပ်စုငယ်များ (Discrete Latent Codes) အဖြစ် အရင်ခွဲခြားထားပါသည်။ ထို့နောက် Transformer က မည်သည့် Latent Code ကို ရွေးချယ်ရမည်ကို ခန့်မှန်းပြီး၊ Residual Head က သေးငယ်သော အတိအကျ ကွာဟချက် (Continuous Offset) ကို ထပ်မံတွက်ချက် ပေါင်းစပ်ပေးပါသည်။
- **အသုံးပြုပုံ**: Multimodal (ရွေးချယ်စရာ လမ်းကြောင်းမျိုးစုံရှိသော) လူသားသရုပ်ပြ ဒေတာများကို တိကျချောမွေ့စွာ သင်ယူရာတွင် သုံးပါသည်။
- **Reference**: Lee et al., *Behavior Generation with Latent Actions* (2024)

---

### ၂၃. WALL-OSS / Wall-X
- **Identifier**: `wall_x` (`WallXPolicy`, `WallXConfig`)
- **Architecture**: Qwen2.5-VL Backbone + Continuous Flow Matching Action Head + LoRA Adaptation
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Vision-Language Prompts၊ Multi-camera Images၊ စက်ရုပ် State
  - **Output**: Cross-embodiment Action Chunks
  - **တွက်ချက်ပုံ**: XSquare Robot မှ ဖန်တီးထားသော ပွင့်လင်းမြင်သာ စက်ရုပ် Foundation Model ဖြစ်ပြီး၊ စက်ရုပ်အမျိုးအစား ပုံစံမတူညီမှုများ (Arms, Mobile Manipulators, Humanoids) ကို Qwen2.5-VL ၏ Multimodal Reasoning နှင့် Flow Matching နည်းပညာဖြင့် တစ်ပြိုင်နက် ချိတ်ဆက်တွက်ထုတ်ပေးပါသည်။
- **အသုံးပြုပုံ**: မတူညီသော စက်ရုပ်ပုံသဏ္ဌာန်များ (Cross-embodiment) ပေါ်တွင် Foundation Model အား LoRA ဖြင့် လွယ်ကူစွာ Post-train လုပ်ရန် သုံးပါသည်။
- **Reference**: XSquare Robot Team, *WALL-OSS: Open-Source Foundation Model for Embodied Intelligence* (2025)

---

### ၂၄. X-VLA
- **Identifier**: `xvla` (`XVLAPolicy`, `XVLAConfig`)
- **Architecture**: Florence-2 VLM Embeddings + Soft-Prompt Learner + Temporal/Action Transformer Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Vision-Language Inputs နှင့် စက်ရုပ် သတ်မှတ်ချက်ဆိုင်ရာ Soft Prompts
  - **Output**: Flow-matching Action Chunks
  - **တွက်ချက်ပုံ**: မတူညီသော စက်ရုပ်များ (Different arm configurations, camera setups, degrees of freedom) ကို "သီးခြား Task တစ်ခုစီ" အဖြစ် သတ်မှတ်ပြီး၊ သင်ယူနိုင်သော **Soft Prompts** များဖြင့် ပုံဖော်ပေးပါသည်။ မော်ဒယ် တစ်ခုတည်းကပင် Soft Prompt ကို ဖတ်ရှုပြီး ထိုစက်ရုပ်၏ စက်ပိုင်းဆိုင်ရာ ကန့်သတ်ချက်နှင့် ကိုက်ညီသော Action များကို ခွဲခြားတွက်ထုတ်ပေးနိုင်ပါသည်။
- **အသုံးပြုပုံ**: မော်ဒယ်တစ်ခုတည်းဖြင့် စက်ရုပ်အမျိုးအစား ရာပေါင်းများစွာကို ကိုင်တွယ်မောင်းနှင်နိုင်စေရန် သုံးပါသည်။
- **Reference**: X-VLA Team, *X-VLA: The First Soft-Prompted Robot Foundation Model for Any Robot, Any Task* (2025)

---

## ၃. Reward Models (ဆုကြေးနှင့် လုပ်ငန်းပြီးမြောက်မှု ခန့်မှန်း မော်ဒယ်များ - ၄ မျိုး)

Policy မော်ဒယ်များအပြင် LeRobot တွင် Reinforcement Learning (RL) နှင့် ဒေတာရွေးချယ်မှုများအတွက် အသုံးပြုသော **Reward Models** ၄ မျိုး ပါဝင်ပါသည်-

### ၁. SARM (Stage-Aware Reward Model)
- **Identifier**: `sarm` (`SARMRewardModel`, `SARMConfig` in `src/lerobot/rewards/sarm/`)
- **Architecture**: CLIP Vision/Text Transformer + Stage Classifier Head + Subtask Progress Regression Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ဗီဒီယိုပြကွက်များ (Video frames) နှင့် သဘာဝဘာသာစကားဖြင့် ရေးသားထားသော Subtask အဆင့်ဆင့်ဖော်ပြချက်များ
  - **Output**:
    1. **Stage Index ($k$)**: လုပ်ငန်း၏ မည်သည့်အဆင့်ကို ရောက်ရှိနေပြီလဲဟူသော Discrete Stage
    2. **Stage Progress ($\tau \in [0, 1]$)**: လက်ရှိအဆင့်အတွင်း ရာခိုင်နှုန်းမည်မျှ ပြီးမြောက်ပြီလဲဟူသော တန်ဖိုး
    3. **Normalized Progress Reward ($[0, 1]$)**: စုစုပေါင်း လုပ်ငန်းတစ်ခုလုံး ပြီးမြောက်ရန် မည်မျှနီးစပ်ပြီလဲဟူသော Dense Reward Signal
  - **အသုံးပြုပုံ**: အဆင့်များစွာပါဝင်သော ရှည်လျားသည့် စက်ရုပ်လုပ်ငန်းများတွင် အဆင့်မြင့် သရုပ်ပြဒေတာများကို အလေးသာစေရန် **Reward-Aligned Behavior Cloning (RA-BC)** တွင် Sample Weights တွက်ချက်ခြင်းနှင့် RL အတွက် Dense Reward အဖြစ် သုံးပါသည်။
  - **Reference**: Chen et al., *SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation* (2025)

---

### ၂. ROBOMETER
- **Identifier**: `robometer` (`RobometerRewardModel`, `RobometerConfig` in `src/lerobot/rewards/robometer/`)
- **Architecture**: Fine-tuned Qwen3-VL-4B-Instruct + Progress Head + Success Head + Preference Head
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: စက်ရုပ်လှုပ်ရှားမှု ဗီဒီယိုလမ်းကြောင်း (Trajectory Video) နှင့် Task Description
  - **Output**:
    1. **Per-frame Task Progress ($[0, 1]$)**: တစ်ဖရိန်ချင်းစီအလိုက် တိုးတက်မှု
    2. **Frame-level Success Probability**: အလုပ်အောင်မြင်မှု ဖြစ်နိုင်ခြေ ရာခိုင်နှုန်း
    3. **Trajectory Preference**: မည်သည့် ဗီဒီယိုလမ်းကြောင်းက အလုပ်ပိုမိုပြီးမြောက်သနည်းဟူသော နှိုင်းယှဉ်ချက်
  - **အသုံးပြုပုံ**: စက်ရုပ်လုပ်ဆောင်မှု ဗီဒီယိုများကို အကဲဖြတ်စစ်ဆေးခြင်းနှင့် General-purpose Reward Signal ထုတ်ယူရန် သုံးပါသည်။
  - **Reference**: Liang et al., *ROBOMETER: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons* (2026)

---

### ၃. TOPReward
- **Identifier**: `topreward` (`TOPRewardModel`, `TOPRewardConfig` in `src/lerobot/rewards/topreward/`)
- **Architecture**: Off-the-shelf VLM (မူလသတ်မှတ်ချက်: Qwen3-VL-8B-Instruct) Log-Likelihood Evaluator
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: စက်ရုပ် ဗီဒီယိုပြကွက်များနှင့် အလုပ်ညွှန်ကြားချက် (ဥပမာ- "Open the drawer")
  - **Output**: Zero-shot Log-probability Reward Signal
  - **တွက်ချက်ပုံ**: မော်ဒယ်ကို သီးသန့် Train ရန် မလိုဘဲ (Zero-shot)၊ VLM ထံသို့ `"အထက်ပါ ဗီဒီယိုသည် ဤအလုပ်ကို အောင်မြင်စွာ ပြီးမြောက်ကြောင်း ပြသနေသည်။ မှန်သလား၊ မှားသလား။ အဖြေမှာ: True"` ဟု မေးခွန်းထုတ်ကာ VLM မှ `True` တိုကင် ထွက်ပေါ်လာမည့် Log-probability ($\log P(\text{True})$) ကို Reward အဖြစ် တိုက်ရိုက်တွက်ထုတ်ယူပါသည်။
  - **အသုံးပြုပုံ**: Model training ထပ်မံမလုပ်ဘဲ Zero-shot ဖြင့် စက်ရုပ်အောင်မြင်မှုကို ချက်ချင်းဆုကြေးတွက်ယူရန် သုံးပါသည်။
  - **Reference**: *TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics* (2026)

---

### ၄. Reward Classifier
- **Identifier**: `reward_classifier` (`Classifier`, `RewardClassifierConfig` in `src/lerobot/rewards/classifier/`)
- **Architecture**: ResNet-10 / CNN သို့မဟုတ် Transformer-based Binary/Multi-class Classifier
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: ကင်မရာပုံရိပ်များ (Multi-camera views)
  - **Output**: Success / Failure Class Probability သို့မဟုတ် Multi-class Classification Logits
  - **တွက်ချက်ပုံ**: စက်ရုပ်လက်ရှိရောက်ရှိနေသော အနေအထားသည် အလုပ်အောင်မြင်သွားပြီလား (Success=1)၊ မအောင်မြင်သေးဘူးလား (Fail=0) ဟူသော ရလဒ်ကို Supervised Learning ဖြင့် တိုက်ရိုက် ခွဲခြားတွက်ချက်ပေးပါသည်။
  - **အသုံးပြုပုံ**: Reinforcement Learning နှင့် စမ်းသပ်စစ်ဆေးမှုများတွင် Sparse/Binary Reward Signal ထုတ်ပေးရန် သုံးပါသည်။

---

## ၄. Reinforcement Learning Algorithm (RL မော်ဒယ်)

### SAC (Soft Actor-Critic)
- **Identifier**: `sac` (`SACAlgorithm`, `SACAlgorithmConfig` in `src/lerobot/rl/algorithms/sac/`)
- **Architecture**: Actor (GaussianActorPolicy) + Double Critic Networks + Temperature $\alpha$ Auto-tuning
- **ဘာအတွက်လဲ / ဘာတွက်တာလဲ (What it computes)**:
  - **Input**: Transitions $(s, a, r, s', d)$ (State, Action, Reward, Next State, Done)
  - **Output**: အမြင့်ဆုံး Reward ရရှိစေမည့် Policy Weights များနှင့် Q-value များ
  - **တွက်ချက်ပုံ**: Maximum Entropy Reinforcement Learning နိယာမအရ Expected Return သာမက Policy ၏ Entropy (မသေချာမှု/စူးစမ်းရှာဖွေနိုင်စွမ်း) ကိုပါ အမြင့်ဆုံးဖြစ်အောင် Bellman Equation ဖြင့် Q-networks များနှင့် Actor Network ကို စဉ်ဆက်မပြတ် Optimize လုပ်ပေးပါသည်။
  - **အသုံးပြုပုံ**: Real-world စက်ရုပ်များတွင် လူသားက ကြားဖြတ်ကူညီပေးသော Human-in-the-loop SERL (HIL-SERL) စနစ်ဖြင့် လျင်မြန်စွာ ကျွမ်းကျင်မှုသင်ယူစေရန် သုံးပါသည်။

---

## ၅. အကျဉ်းချုပ် ဇယား (Comparison Summary Table)

| စဉ် | Model အမည် | Identifier | အဓိက နည်းပညာ / Backbone | ဘာတွက်တာလဲ (Primary Output & Computation) |
|---|---|---|---|---|
| ၁ | **ACT** | `act` | CVAE + Transformer | အနာဂတ် Action Chunks အစီအစဉ် |
| ၂ | **Diffusion Policy** | `diffusion` | 1D CNN / Transformer + DDPM | Noise မှ Action Trajectory သို့ အဆင့်ဆင့် Denoise လုပ်ခြင်း |
| ၃ | **DM05** | `dm05` | Dexmal VLA + Flow Matching | Open-world Action Chunks |
| ၄ | **EO-1** | `eo1` | Qwen2.5-VL + Flow Matching | ရုပ်ပုံ/စာသား စကားဝိုင်းပုံစံမှ Action Chunks |
| ၅ | **Evo-1** | `evo1` | InternVL3 + Flow Matching | Multimodal Token မှ Action Chunks |
| ၆ | **FastWAM** | `fastwam` | Wan2.2 Video Stack | Video World Model အခြေခံ Direct Action Chunks |
| ၇ | **FineART-VLA** | `fineart_vla` | π₀.₅ (Dual-System) | Language Subtask (System 2) + Action Chunks (System 1) |
| ၈ | **FLUX 3 Action** | `flux3` | FLUX.3 Diffusion Transformer | Joint Video Latents နှင့် Action Denoising |
| ၉ | **Gaussian Actor** | `gaussian_actor` | CNN/ResNet + Gaussian MLP | Continuous Action Mean ($\mu$) နှင့် Std ($\sigma$) |
| ၁၀ | **GR00T N1.7** | `groot` | Cosmos-Reason2 / Qwen3-VL | လူတူစက်ရုပ် (Humanoid) Cross-embodiment Action Chunks |
| ၁၁ | **LaWAM** | `lawam` | Qwen3-VL + DINOv3 LAM | Latent World Dynamics နှင့် Action Chunks |
| ၁၂ | **LingBot-VA** | `lingbot_va` | Wan2.2 Dual-Stream MoT | အနာဂတ် ဗီဒီယိုနှင့် Actions တပြိုင်နက် Autoregressive တွက်ချက်ခြင်း |
| ၁၃ | **MolmoAct2** | `molmoact2` | Molmo VLM + Layerwise Flow Expert | ရုပ်ပုံ/စာသားမှ Continuous Action Chunks |
| ၁၄ | **Multi-Task DiT** | `multi_task_dit` | CLIP + Diffusion Transformer | လုပ်ငန်းစုံအတွက် Generative Action Trajectory |
| ၁၅ | **π₀ (Pi0)** | `pi0` | PaliGemma + Flow Action Expert | Generalist VLA High-Frequency Action Chunks |
| ၁၆ | **π₀-FAST** | `pi0_fast` | Gemma + FAST Tokenizer | FAST တိုကင်များဖြင့် Autoregressive Action Prediction |
| ၁၇ | **π₀.₅ (Pi0.5)** | `pi05` | π₀ အဆင့်မြှင့် + Flow + RTC | မမြင်ဖူးသော ပတ်ဝန်းကျင်သစ်များအတွက် Generalized Actions |
| ၁၈ | **RTC** | `rtc` | Real-Time Chunking Engine | Model Latency ကို ချေဖျက်ပေးသော Action Interpolation |
| ၁၉ | **SmolVLA** | `smolvla` | SmolLM/Vision + Flow Expert | အရင်းအမြစ်နည်းပါးသော GPU များအတွက်ပေါ့ပါးသည့် VLA Actions |
| ၂၀ | **TD-MPC** | `tdmpc` | Latent Dynamics + MPPI/CEM | Latent Trajectory Optimization ဖြင့် အကောင်းဆုံး Action |
| ၂၁ | **VLA-JEPA** | `vla_jepa` | Qwen3-VL + V-JEPA2 + DiT | Non-generative World Features အခြေခံ Action Chunks |
| ၂၂ | **VQ-BeT** | `vqbet` | VQ-VAE + minGPT | Latent Action Codes + Residual Continuous Offset |
| ၂၃ | **WALL-OSS** | `wall_x` | Qwen2.5-VL + Flow Matching | စက်ရုပ်အမျိုးအစားစုံ (Cross-embodiment) Action Chunks |
| ၂၄ | **X-VLA** | `xvla` | Florence-2 + Soft Prompts | Soft Prompts အခြေခံ မည်သည့်စက်ရုပ်အတွက်မဆို Actions |
| ၂၅ | **SARM** | `sarm` | CLIP + Dual Stage/Progress Heads | Subtask Stage Index နှင့် Normalized Progress Reward |
| ၂၆ | **ROBOMETER** | `robometer` | Qwen3-VL-4B Multi-head | Frame-level Progress, Success Prob, Preference |
| ၂၇ | **TOPReward** | `topreward` | Zero-shot Qwen3-VL | Task ပြီးမြောက်မှုဆိုင်ရာ VLM Token Log-probability |
| ၂၈ | **Reward Classifier** | `reward_classifier` | ResNet-10 / CNN / Transformer | အလုပ် အောင်မြင်/မအောင်မြင် Binary/Multi-class စစ်ဆေးခြင်း |
| ၂၉ | **SAC** | `sac` | Soft Actor-Critic RL Engine | Expected Return နှင့် Entropy အမြင့်ဆုံးရမည့် RL Updates |
