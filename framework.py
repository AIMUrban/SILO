import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM

from tools import init_tokenizer


class MyModel(nn.Module):
    def __init__(self, args, device, config, mode):
        super().__init__()
        self.device = device
        self.args = args
        self.config = config
        self.mode = mode

        self.num_user = config.Dataset.num_users
        self.num_location = config.Dataset.num_locations
        self.num_time = 28

        self.llm = MyLLM(args=args, device=device)

        token_hidden_size = self.llm.model.config.hidden_size
        user_id_emb_size = args.user_dim
        loc_id_emb_size = args.loc_dim
        time_id_emb_size = args.time_dim

        
        self.cat_text = [
                "Residential: Locations where people typically spend weekday mornings and evenings, with longer stays on weekends.",
                "Work: Locations primarily active during weekday mornings and afternoons, with limited activity on weekends.",
                "Leisure/Recreation: Locations often visited on weekday afternoons, extending into mornings and evenings on weekends.",
                "Entertainment: Locations frequently attended on Friday evenings and weekend nights, with fewer weekday morning visits.",
                "Shopping/Commercial: Locations often visited during weekday afternoons and evenings, shifting to weekend mornings and afternoons.",
                "Education: Locations attended mainly during weekday mornings and afternoons, with little to no presence on weekends.",
                "Healthcare: Locations usually visited on weekday mornings and afternoons, with occasional visits on weekend mornings.",
                "Transportation: Locations busier on weekday mornings and evenings, with more varied patterns on weekends.",
                "Public/Social: Locations visited across all periods on both weekdays and weekends, though intensity varies."
            ]
        
        self.profile_text = [
                "Early Bird: Users who are active in the morning on weekdays, shifting to the afternoon on weekends.",
                "Daytime Dweller: Users who remain primarily active during the afternoon across all days.",
                "Evening Enthusiast: Users who are most active during the evening, often showing a significant increase on weekends.",
                "Night Owl: Users who are active at night—whether for work, study, or leisure—across both weekdays and weekends.",
                "Weekday Regular: Users who consistently engage in morning and evening activities on weekdays, with reduced activity on weekends.",
                "All-Day Active: Users who are frequently active throughout the day, from morning to evening, reflecting a busy schedule."
            ]


        self.fuse_lt = nn.Sequential(
            nn.Linear(token_hidden_size * 2, token_hidden_size),
            nn.LeakyReLU(0.2),
        )

        self.category_embeddings = self.compute_category_embeddings()

        self.query_proj = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 4),
            nn.Linear(token_hidden_size // 4, token_hidden_size)
        )
        self.key_proj = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 4),
            nn.Linear(token_hidden_size // 4, token_hidden_size)
        )
        self.value_proj = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 4),
            nn.Linear(token_hidden_size // 4, token_hidden_size)
        )
        self.update_cat_text = nn.Linear(token_hidden_size, token_hidden_size)


        self.time_emb_str = self.compute_text_time_embeddings()

        loc_sem_embeddings = torch.load(
            open(f'{args.dataset}/{args.llm}/loc_sem_embeddings_{args.numf_l}f.pt', 'rb'), map_location='cpu').float()
        self.loc_emb_str = loc_sem_embeddings

        user_sem_embeddings = torch.load(
            open(f'{args.dataset}/{args.llm}/user_sem_embeddings_{args.numf_l}f.pt', 'rb'), map_location='cpu').float()
        self.user_emb_str = user_sem_embeddings

        # project to token space
        self.str_loc_proj = nn.Sequential(
                nn.Linear(token_hidden_size, token_hidden_size // 4),
                nn.Linear(token_hidden_size // 4, token_hidden_size)
            )
        self.str_time_proj = nn.Sequential(
                nn.Linear(token_hidden_size, token_hidden_size // 4),
                nn.Linear(token_hidden_size // 4, token_hidden_size)
            )
        self.str_user_proj = nn.Sequential(
                nn.Linear(token_hidden_size, token_hidden_size // 4),
                nn.Linear(token_hidden_size // 4, token_hidden_size)
            )

        # different modal may need to be fused ?
        self.fuse_loc = nn.Sequential(
            nn.Linear(token_hidden_size+loc_id_emb_size, token_hidden_size),
            nn.LeakyReLU(0.2),
        )
        self.fuse_time = nn.Sequential(
            nn.Linear(token_hidden_size+time_id_emb_size, token_hidden_size),
            nn.LeakyReLU(0.2),
        )

        self.loc_emb = nn.Embedding(self.num_location, loc_id_emb_size)
        self.time_emb = nn.Embedding(self.num_time, time_id_emb_size)

        self.user_emb = nn.Embedding(self.num_user, user_id_emb_size)

        self.profile_embeddings = self.compute_profile_embeddings()

        self.profile_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_hidden_size, token_hidden_size // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(token_hidden_size // 2, token_hidden_size)
            )
            for _ in range(len(self.profile_text))
        )
        self.update_profile_text = nn.Linear(token_hidden_size, token_hidden_size)


        input_dim = token_hidden_size
        input_dim += user_id_emb_size

        self.loc_header = nn.Sequential(
            nn.Linear(input_dim, token_hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(token_hidden_size//2, self.num_location),
        )

        self.to(self.device)

    @torch.no_grad()
    def compute_text_user_embeddings(self):
        user_text = [f'user_{i}' for i in range(self.num_user)]
        embeddings = []
        for user in user_text:
            tokens = self.llm.tokenizer(user, return_tensors="pt", add_special_tokens=False).to(self.device)
            outputs = self.llm.model(**tokens, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0).detach().float()
    
    @torch.no_grad()
    def compute_text_time_embeddings(self):
        weekday_text = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        period_text = ['morning', 'afternoon', 'evening', 'night']
        embeddings = []
        for w in weekday_text:
            for p in period_text:
                tokens = self.llm.tokenizer(f'{w} {p}', return_tensors="pt", add_special_tokens=False).to(self.device)
                outputs = self.llm.model(**tokens, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1)
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0).detach().float()
    
    @torch.no_grad()
    def compute_category_embeddings(self):
        embeddings = []
        for desc in self.cat_text:
            tokens = self.llm.tokenizer(desc, return_tensors="pt", add_special_tokens=False).to(self.device)
            outputs = self.llm.model(**tokens, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0).detach().float()

    @torch.no_grad()
    def compute_profile_embeddings(self):
        embeddings = []
        for desc in self.profile_text:
            tokens = self.llm.tokenizer(desc, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                outputs = self.llm.model(**tokens, output_hidden_states=True)
                embedding = outputs['hidden_states'][-1].mean(dim=1)
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0).detach().float()

    def forward(self, batch_data):
        user = torch.tensor(batch_data['user'], dtype=torch.int)
        loc_his = torch.tensor(np.array(batch_data['loc_his']), dtype=torch.long, device=self.device)
        loc_cur = torch.tensor(batch_data['loc_cur'], dtype=torch.long, device=self.device)
        timeslot_his = torch.tensor(batch_data['timeslot_his'], dtype=torch.int, device=self.device)
        timeslot_cur = torch.tensor(batch_data['timeslot_cur'], dtype=torch.int, device=self.device)
        samples = {}

        loc_emb_id = self.loc_emb(torch.arange(end=self.num_location, dtype=torch.int, device=self.device))
        time_emb_id = self.time_emb(torch.arange(end=self.num_time, dtype=torch.int, device=self.device))

        loc_emb_str = self.loc_emb_str
        time_emb_str = self.time_emb_str
        loc_emb_str = self.str_loc_proj(loc_emb_str.to(self.device))
        time_emb_str = self.str_time_proj(time_emb_str)
        loc_emb_fusion = self.fuse_loc(torch.cat([loc_emb_id, loc_emb_str], dim=-1))
        time_emb_fusion = self.fuse_time(torch.cat([time_emb_id, time_emb_str], dim=-1))

        category_embeddings = self.category_embeddings
        category_embeddings = self.update_cat_text(category_embeddings)
        q = self.query_proj(loc_emb_fusion)
        k = self.key_proj(category_embeddings)
        v = self.value_proj(category_embeddings)
        attn_scores = torch.matmul(q, k.T)/(k.size(1) ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        loc_cat_emb = torch.matmul(attn_scores, v)
        loc_emb_fusion = loc_emb_fusion + loc_cat_emb

        his_emb_loc = loc_emb_fusion[loc_his]
        cur_emb_loc = loc_emb_fusion[loc_cur]
        his_emb_time = time_emb_fusion[timeslot_his]
        cur_emb_time = time_emb_fusion[timeslot_cur]
        his_emb = self.fuse_lt(torch.cat([his_emb_loc, his_emb_time], dim=-1))
        cur_emb = self.fuse_lt(torch.cat([cur_emb_loc, cur_emb_time], dim=-1))

        samples.update({
            'his_emb': his_emb,
            'cur_emb': cur_emb,
        })
        
        samples.update(batch_data)
        samples.update({
            'user_emb_str': self.str_user_proj(self.user_emb_str[user].to(self.device)),
        })

        seq_output, user_sem_emb = self.llm(samples)

        user_sem_emb = user_sem_emb.float()
        seq_output = seq_output.float()

        user_emb_id = self.user_emb(user.to(self.device))
        seq_output_comb = torch.cat([seq_output, user_emb_id], dim=-1)
        logits = self.loc_header(seq_output_comb)

        profile_emb = self.profile_embeddings
        profile_emb = self.update_profile_text(profile_emb)
        profile_emb_norm = F.normalize(profile_emb, dim=1)
        user_sem_emb_norm = F.normalize(user_sem_emb, dim=1)
        cos_similarity = torch.matmul(user_sem_emb_norm, profile_emb_norm.T)
        gating_weights = F.softmax(cos_similarity, dim=1)

        expert_outputs = torch.stack([expert(profile_emb[i]) for i, expert in enumerate(self.profile_experts)], dim=0)
        expert_logits = torch.matmul(expert_outputs, loc_cat_emb.T)

        weighted_expert_logits = torch.matmul(gating_weights, expert_logits)

        adjusted_logits = (logits + weighted_expert_logits)
        loc_output = F.log_softmax(adjusted_logits, dim=-1)

        return loc_output



class MyLLM(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model_id = args.llm_id
        self.device = device

        llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        self.tokenizer = init_tokenizer(self.model_id, additional_tokens=['[UserEmb]'])   


        # print(llm_model)
        if args.num_layer:
            if args.llm == 'gpt2':
                llm_model.transformer.h = llm_model.transformer.h[:args.num_layer]
            elif 'llama' in args.llm:
                llm_model.model.layers = llm_model.model.layers[:args.num_layer]

        # print(llm_model)
        for _, param in llm_model.named_parameters():
            param.requires_grad = False

        if args.llm == 'gpt2':
            target_modules = ['c_attn']
            fan_in_fan_out = True
        else:
            target_modules = ["q_proj", "k_proj"]
            fan_in_fan_out = False
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            fan_in_fan_out=fan_in_fan_out,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05
        )
        llm_model = get_peft_model(llm_model, lora_config)

        llm_model.resize_token_embeddings(len(self.tokenizer))
        self.model = llm_model
        self.tokenizer.padding_side = 'left'

    def forward(self, x):
        batch_size = x['his_emb'].size(0)
        user = x['user']
        prompts = []

        for idx in range(batch_size):
            prompt = (
                f"Your task is to predict the next location for user_{user[idx]}, based on the given visitation sequence. "
                "Additional requirements: "
                "1. Incorporate the frequency data of the user's historical activity times, and summarized it into one word with the sequence. "
                f"The frequency data of user_{user[idx]} is:"
            )
            prompts.append(prompt)

        text_input_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        prompts_embed = self.model.get_input_embeddings()(text_input_tokens['input_ids'].to(self.device))

        prompt2 = "The sequence is:"
        text_input_tokens2 = self.tokenizer(
            prompt2,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompts_embed2 = self.model.get_input_embeddings()(text_input_tokens2['input_ids'].to(self.device))
        prompts_embed2 = prompts_embed2.repeat(batch_size, 1, 1)
        user_emb_str = x['user_emb_str'].unsqueeze(1)

        prompt3 = "The summarized word is: '[UserEmb]'."
        text_input_tokens3 = self.tokenizer(
            prompt3,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompts_embed3 = self.model.get_input_embeddings()(text_input_tokens3['input_ids'].to(self.device))
        prompts_embed3 = prompts_embed3.repeat(batch_size, 1, 1)

        input_embeds = torch.cat(
            [prompts_embed, user_emb_str, prompts_embed2, x['his_emb'], x['cur_emb'], prompts_embed3], dim=1)
        input_embeds = input_embeds.to(prompts_embed.dtype)
        output = self.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        last_hidden_states = output['hidden_states'][-1]
        seq_output = last_hidden_states[:, -prompts_embed3.size(1) - 1, :]

        user_token_id = self.tokenizer.convert_tokens_to_ids("[UserEmb]")
        emb_token_position = (text_input_tokens3['input_ids'][0] == user_token_id).nonzero(as_tuple=True)[0].item()
        final_position = prompts_embed3.size(1) - emb_token_position
        user_sem_emb = last_hidden_states[:, -final_position, :]

        return seq_output, user_sem_emb
