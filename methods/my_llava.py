from transformers import LlavaForConditionalGeneration
from transformers.models.llava.modeling_llava import ModelOutput, is_torchdynamo_compiling # type: ignore
from dataclasses import dataclass

import torch
from typing import List, Optional, Tuple, Union
from torch import nn

import ipdb
from methods.my_llama import MyLlamaForCausalLM
from transformers import AutoModelForCausalLM


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    token_select: Optional[torch.FloatTensor] = None


def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class TokenSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer=1, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        # x [2, 1024, 4096])
        b, l = x.shape[:2]

        # ipdb.set_trace()

        logits = self.mlp_head(x[:, 1:-1, :])
        
        token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        token_select = torch.cat([token_select.new_ones(b, 1, 1), token_select, token_select.new_ones(b, 1, 1)], dim=1)  # cls
        
        return token_select, logits


def extract_subsequence_mask(input_ids, start_token=32000, end_token=32001):
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # 找最后一个 start_token
        start_pos = (seq == start_token).nonzero(as_tuple=False).flatten()
        if len(start_pos) == 0:
            continue  # no start token found
        start_idx = start_pos[-1].item()

        # 找第一个 end_token 且在 start_token 之后
        end_pos = (seq[start_idx + 1:] == end_token).nonzero(as_tuple=False)
        if len(end_pos) == 0:
            continue  # no end token after start token
        end_idx = start_idx + 1 + end_pos[0].item()

        # 构造 mask
        mask[i, start_idx + 1 : end_idx] = 1  # 包含结束 token
        
        mask[i, :start_idx + 1] = 2
        mask[i, end_idx:] = 2

    return mask


IMAGE_TOKEN_INDEX = 32000
PAD_TOKEN_INDEX = 32001






from transformers import LlamaForCausalLM, AutoTokenizer, AddedToken

class MyLlava(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        hidden_size = config.text_config.hidden_size
        self.mlp_token_select = TokenSelect(dim_in=hidden_size, num_sub_layer=1)
        
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        # ipdb.set_trace()
        # self.language_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", config=config.text_config) # 



        # tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        # add_num = 1 
        # ## 添加 <image>
        # tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
        
        # # 首先计算原始词表的均值和方差
        # pre_expansion_embeddings = self.language_model.language_model.model.embed_tokens.weight.data
        # mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        # n = pre_expansion_embeddings.size()[0]
        # sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        # dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # # 扩充词表
        # pad_shape = 64
        # vocab_size = config.text_config.vocab_size
        # # 第二个参数pad_shape 参数名叫做 pad to multiple of , 意思是直接弄到 64 的某一个倍数
        # self.language_model.resize_token_embeddings(config.text_config.vocab_size + add_num , pad_shape)
        # self.language_model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        #     tuple((dist.sample() for _ in range(self.language_model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))),
        #     dim=0,
        # )
        # self.language_model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        #     tuple((dist.sample() for _ in range(self.language_model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        #     dim=0,
        # )


    
    # def get_image_features(
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     vision_feature_layer: Union[int, List[int]],
    #     vision_feature_select_strategy: str,
    #     **kwargs,
    # ):
    #     """
    #     Obtains image last hidden states from the vision tower and apply multimodal projection.

    #     Args:
    #         pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
    #            The tensors corresponding to the input images.
    #         vision_feature_layer (`Union[int, List[int]]`):
    #             The index of the layer to select the vision feature. If multiple indices are provided,
    #             the vision feature of the corresponding indices will be concatenated to form the
    #             vision features.
    #         vision_feature_select_strategy (`str`):
    #             The feature selection strategy used to select the vision feature from the vision backbone.
    #             Can be one of `"default"` or `"full"`
    #     Returns:
    #         image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    #     """
    #     if vision_feature_select_strategy not in ["default", "full"]:
    #         raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

    #     kwargs = {k: v for k, v in kwargs.items() if v is not None}
    #     # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
    #     image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

    #     # If we have one vision feature layer, return the corresponding hidden states,
    #     # otherwise, select the hidden states of each feature layer and concatenate them
    #     if isinstance(vision_feature_layer, int):
    #         selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    #         if vision_feature_select_strategy == "default":
    #             selected_image_feature = selected_image_feature[:, 1:]
    #     else:
    #         hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
    #         # For default; crop CLS from each hidden state in the hidden state pool
    #         if vision_feature_select_strategy == "default":
    #             hs_pool = [hs[:, 1:] for hs in hs_pool]
    #         selected_image_feature = torch.cat(hs_pool, dim=-1)

    #     image_features = self.multi_modal_projector(selected_image_feature)
    #     return image_features
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = -2,
        vision_feature_select_strategy: Optional[str] = "default",
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        **lm_kwargs,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # step 1: 获取模型学习的可学习 mask
            policy_token = inputs_embeds
            sub_token_select, _ = self.mlp_token_select(policy_token)
            
            # step 2: 基于 input_ids，提取合法片段（1 表示是 text token 区域）
            origin_text_mask = extract_subsequence_mask(input_ids, IMAGE_TOKEN_INDEX, PAD_TOKEN_INDEX)
            final_mask = torch.where(
                origin_text_mask.unsqueeze(-1) == 2,
                torch.ones_like(sub_token_select),  # 位置为2，设为1
                origin_text_mask.unsqueeze(-1) * sub_token_select    # 其它情况，做乘法
            ).squeeze(-1)
            # ipdb.set_trace()
            masked_inputs_ids = torch.where(final_mask.squeeze(-1) == 1, input_ids, torch.full_like(input_ids, 396))

            masked_inputs_embeds = self.get_input_embeddings()(masked_inputs_ids)
            # inputs_embeds = inputs_embeds + masked_inputs_embeds
            inputs_embeds = inputs_embeds
            
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)


        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        
        logits = outputs[0]
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            token_select=sub_token_select,
        )