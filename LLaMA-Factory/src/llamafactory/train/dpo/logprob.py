# src/llamafactory/train/dpo/logprob.py

import torch
from typing import Dict, List, Optional, Union, Tuple
from transformers import TrainingArguments, PreTrainedModel
from ...hparams import ModelArguments, DataArguments, FinetuningArguments
from ...model import load_model, load_tokenizer
from ...data import get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from ...extras.constants import IGNORE_INDEX
from .trainer import CustomDPOTrainer


def adjust_batch_padding(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Ensure that the attention_mask has the same sequence length as input_ids.
    If it is shorter, pad it on the right with zeros.
    """
    if "input_ids" in batch and "attention_mask" in batch:
        seq_len = batch["input_ids"].size(1)
        current_len = batch["attention_mask"].size(1)
        if current_len < seq_len:
            pad_length = seq_len - current_len
            pad_tensor = torch.zeros(
                batch["attention_mask"].size(0),
                pad_length,
                dtype=batch["attention_mask"].dtype,
                device=batch["attention_mask"].device,
            )
            batch["attention_mask"] = torch.cat([batch["attention_mask"], pad_tensor], dim=1)
    return batch


class LogProbabilityCalculator:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        finetuning_args: Optional[FinetuningArguments] = None,
        model: Optional[PreTrainedModel] = None,
        ref_model: Optional[PreTrainedModel] = None,
        ref_model_args: Optional[ModelArguments] = None,
        beta: float = 0.1,
        **kwargs
    ):
        self.model_args = model_args
        self.data_args = data_args or DataArguments()
        self.training_args = training_args or TrainingArguments(
            output_dir="./logprob_outputs",
            remove_unused_columns=False,
            per_device_eval_batch_size=1
        )
        self.finetuning_args = finetuning_args or FinetuningArguments()
        self.beta = beta  # Beta parameter for reward scaling
        
        # Set training args to disable training and evaluation
        self.training_args.do_train = False
        self.training_args.do_eval = False
        
        # Load tokenizer module
        self.tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = self.tokenizer_module["tokenizer"]
        
        # Load model (policy model)
        self.model = model or load_model(
            self.tokenizer,
            self.model_args,
            self.finetuning_args,
            is_trainable=False
        )
        
        # Load reference model if provided
        self.ref_model = None
        if ref_model is not None:
            self.ref_model = ref_model
        elif ref_model_args is not None:
            # Load reference model with separate model args
            self.ref_model = load_model(
                self.tokenizer,
                ref_model_args,
                self.finetuning_args,
                is_trainable=False
            )
        
        # Setup template
        template_name = self.data_args.template if hasattr(self.data_args, 'template') else "alpaca"
        self.data_args.template = template_name
        self.template = get_template_and_fix_tokenizer(
            self.tokenizer,
            self.data_args
        )
        
        # Create data collator (without multimodal fields)
        self.data_collator = PairwiseDataCollatorWithPadding(
            model=self.model,
            padding=True,
            template=self.template,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            **self.tokenizer_module  # Pass all tokenizer module items
        )
        
        # Create trainer for policy model
        self.trainer = CustomDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,  # Pass reference model to trainer
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            data_collator=self.data_collator,
            callbacks=None, 
            compute_metrics=None,
            **self.tokenizer_module  # Pass all tokenizer module items
        )

    def prepare_pair_inputs(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str]
    ) -> List[Dict]:
        """
        Prepare inputs for pairs of chosen and rejected responses.
        """
        inputs = []
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            # Encode chosen prompt and response pairs
            chosen_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ]
            chosen_encoded = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                messages=chosen_messages
            )
            
            # Encode rejected prompt and response pairs
            rejected_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
            rejected_encoded = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                messages=rejected_messages
            )
            
            # Create features dictionary
            features = {
                "chosen_input_ids": chosen_encoded[0],
                "chosen_attention_mask": chosen_encoded[1],
                "chosen_labels": chosen_encoded[0],  # Use input_ids as labels
                "rejected_input_ids": rejected_encoded[0],
                "rejected_attention_mask": rejected_encoded[1],
                "rejected_labels": rejected_encoded[0],  # Use input_ids as labels
                "images": [],
                "videos": [],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
            inputs.append(features)
        return inputs

    def prepare_inputs(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> List[Dict]:
        """
        Prepare inputs in the format expected by the data collator.
        Multimodal fields have been removed.
        """
        inputs = []
        for prompt, response in zip(prompts, responses):
            # Encode prompt and response pairs
            chosen_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            # Using the same encoding for rejected messages
            chosen_encoded = self.template.encode_oneturn(
                tokenizer=self.tokenizer,
                messages=chosen_messages
            )
            # Create features dictionary without multimodal fields
            features = {
                "chosen_input_ids": chosen_encoded[0],
                "chosen_attention_mask": chosen_encoded[1],
                "chosen_labels": chosen_encoded[0],  # Use input_ids as labels
                "rejected_input_ids": chosen_encoded[0],
                "rejected_attention_mask": chosen_encoded[1],
                "rejected_labels": chosen_encoded[0],  # Use input_ids as labels
                "input_ids": chosen_encoded[0],
                "attention_mask": chosen_encoded[1],
                "labels": chosen_encoded[0],
                "images": [],
                "videos": [],
                "prompt": prompt,
                "chosen": response,
                "rejected": response
            }
            inputs.append(features)
        return inputs

    def get_logprobs(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate log probabilities for a list of prompts and responses.
        """
        # Convert single inputs to lists if necessary
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]
            
        assert len(prompts) == len(responses), "Number of prompts and responses must match"
        
        all_logprobs = []
        all_avg_logprobs = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # Prepare inputs and use data collator
            inputs = self.prepare_inputs(batch_prompts, batch_responses)
            batch = self.data_collator(inputs)
            
            # Adjust batch padding (ensure attention_mask length matches input_ids)
            batch = adjust_batch_padding(batch)
            
            # Move batch to the device
            batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Calculate log probabilities
            with torch.no_grad():
                policy_logps, _, _, _, policy_avg_logps = self.trainer.concatenated_forward(
                    self.trainer.model,
                    batch
                )
                all_logprobs.extend(policy_logps.cpu().tolist())
                all_avg_logprobs.extend(policy_avg_logps.cpu().tolist())

        return {
            "total_logprobs": all_logprobs,
            "avg_logprobs": all_avg_logprobs
        }

    def get_logprobs_with_reference(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate log probabilities for both policy and reference models.
        """
        if self.ref_model is None:
            raise ValueError("Reference model is not provided. Unable to calculate reference log probabilities.")
        
        # Get policy model log probabilities
        policy_logprobs = self.get_logprobs(prompts, responses, batch_size)
        
        all_ref_logprobs = []
        all_ref_avg_logprobs = []

        # Convert single inputs to lists if necessary
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(responses, str):
            responses = [responses]
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # Prepare inputs and use data collator
            inputs = self.prepare_inputs(batch_prompts, batch_responses)
            batch = self.data_collator(inputs)
            
            # Adjust batch padding
            batch = adjust_batch_padding(batch)
            
            # Move batch to the device
            batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Calculate reference log probabilities
            with torch.no_grad():
                ref_logps, _, _, _, ref_avg_logps = self.trainer.concatenated_forward(
                    self.ref_model,
                    batch
                )
                all_ref_logprobs.extend(ref_logps.cpu().tolist())
                all_ref_avg_logprobs.extend(ref_avg_logps.cpu().tolist())

        return {
            "policy_total_logprobs": policy_logprobs["total_logprobs"],
            "policy_avg_logprobs": policy_logprobs["avg_logprobs"],
            "ref_total_logprobs": all_ref_logprobs,
            "ref_avg_logprobs": all_ref_avg_logprobs
        }

    def get_paired_logprobs(
        self,
        prompts: Union[str, List[str]],
        chosen_responses: Union[str, List[str]],
        rejected_responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate log probabilities for chosen and rejected responses.
        """
        # Convert single inputs to lists if necessary
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(chosen_responses, str):
            chosen_responses = [chosen_responses]
        if isinstance(rejected_responses, str):
            rejected_responses = [rejected_responses]
            
        assert len(prompts) == len(chosen_responses) == len(rejected_responses), "Number of prompts and responses must match"
        
        all_chosen_logprobs = []
        all_chosen_avg_logprobs = []
        all_rejected_logprobs = []
        all_rejected_avg_logprobs = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_chosen = chosen_responses[i:i + batch_size]
            batch_rejected = rejected_responses[i:i + batch_size]
            
            # Prepare inputs for pairs
            inputs = self.prepare_pair_inputs(batch_prompts, batch_chosen, batch_rejected)
            batch = self.data_collator(inputs)
            
            # Adjust batch padding
            batch = adjust_batch_padding(batch)
            
            # Move batch to the device
            batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Calculate log probabilities for both chosen and rejected
            with torch.no_grad():
                policy_chosen_logps, policy_rejected_logps, _, _, policy_chosen_avg_logps, policy_rejected_avg_logps = self.trainer.paired_forward(
                    self.trainer.model,
                    batch
                )
                all_chosen_logprobs.extend(policy_chosen_logps.cpu().tolist())
                all_chosen_avg_logprobs.extend(policy_chosen_avg_logps.cpu().tolist())
                all_rejected_logprobs.extend(policy_rejected_logps.cpu().tolist())
                all_rejected_avg_logprobs.extend(policy_rejected_avg_logps.cpu().tolist())

        return {
            "chosen_total_logprobs": all_chosen_logprobs,
            "chosen_avg_logprobs": all_chosen_avg_logprobs,
            "rejected_total_logprobs": all_rejected_logprobs,
            "rejected_avg_logprobs": all_rejected_avg_logprobs
        }

    def get_paired_logprobs_with_reference(
        self,
        prompts: Union[str, List[str]],
        chosen_responses: Union[str, List[str]],
        rejected_responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate log probabilities for chosen and rejected responses using both policy and reference models.
        """
        if self.ref_model is None:
            raise ValueError("Reference model is not provided. Unable to calculate reference log probabilities.")
        
        # Get policy model log probabilities for pairs
        policy_logprobs = self.get_paired_logprobs(prompts, chosen_responses, rejected_responses, batch_size)
        
        all_ref_chosen_logprobs = []
        all_ref_chosen_avg_logprobs = []
        all_ref_rejected_logprobs = []
        all_ref_rejected_avg_logprobs = []

        # Convert single inputs to lists if necessary
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(chosen_responses, str):
            chosen_responses = [chosen_responses]
        if isinstance(rejected_responses, str):
            rejected_responses = [rejected_responses]
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_chosen = chosen_responses[i:i + batch_size]
            batch_rejected = rejected_responses[i:i + batch_size]
            
            # Prepare inputs for pairs
            inputs = self.prepare_pair_inputs(batch_prompts, batch_chosen, batch_rejected)
            batch = self.data_collator(inputs)
            
            # Adjust batch padding
            batch = adjust_batch_padding(batch)
            
            # Move batch to the device
            batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Calculate reference log probabilities for both chosen and rejected
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _, _, ref_chosen_avg_logps, ref_rejected_avg_logps = self.trainer.paired_forward(
                    self.ref_model,
                    batch
                )
                all_ref_chosen_logprobs.extend(ref_chosen_logps.cpu().tolist())
                all_ref_chosen_avg_logprobs.extend(ref_chosen_avg_logps.cpu().tolist())
                all_ref_rejected_logprobs.extend(ref_rejected_logps.cpu().tolist())
                all_ref_rejected_avg_logprobs.extend(ref_rejected_avg_logps.cpu().tolist())

        return {
            "policy_chosen_total_logprobs": policy_logprobs["chosen_total_logprobs"],
            "policy_chosen_avg_logprobs": policy_logprobs["chosen_avg_logprobs"],
            "policy_rejected_total_logprobs": policy_logprobs["rejected_total_logprobs"],
            "policy_rejected_avg_logprobs": policy_logprobs["rejected_avg_logprobs"],
            "ref_chosen_total_logprobs": all_ref_chosen_logprobs,
            "ref_chosen_avg_logprobs": all_ref_chosen_avg_logprobs,
            "ref_rejected_total_logprobs": all_ref_rejected_logprobs,
            "ref_rejected_avg_logprobs": all_ref_rejected_avg_logprobs
        }
    
    def calculate_rewards(
        self, 
        policy_logprobs: List[float], 
        ref_logprobs: List[float]
    ) -> List[float]:
        """
        Calculate DPO rewards using the formula: beta * (policy_logprob - ref_logprob)
        """
        policy_tensor = torch.tensor(policy_logprobs)
        ref_tensor = torch.tensor(ref_logprobs)
        
        rewards = self.beta * (policy_tensor - ref_tensor)
        return rewards.tolist()
    
    def get_rewards(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate log probabilities and rewards for responses using total log probabilities.
        """
        # Get log probabilities from both models
        logprobs = self.get_logprobs_with_reference(prompts, responses, batch_size)
        
        # Calculate rewards using total log probabilities
        total_rewards = self.calculate_rewards(
            logprobs["policy_total_logprobs"],
            logprobs["ref_total_logprobs"]
        )
        
        # Also calculate average rewards for comparison
        avg_rewards = self.calculate_rewards(
            logprobs["policy_avg_logprobs"],
            logprobs["ref_avg_logprobs"]
        )
        
        return {
            "policy_total_logprobs": logprobs["policy_total_logprobs"],
            "policy_avg_logprobs": logprobs["policy_avg_logprobs"],
            "ref_total_logprobs": logprobs["ref_total_logprobs"],
            "ref_avg_logprobs": logprobs["ref_avg_logprobs"],
            "total_rewards": total_rewards,
            "avg_rewards": avg_rewards,
            # Use total_rewards as the default rewards
            "rewards": total_rewards
        }
    
    def get_paired_rewards(
        self,
        prompts: Union[str, List[str]],
        chosen_responses: Union[str, List[str]],
        rejected_responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate rewards for chosen and rejected responses using total log probabilities.
        """
        # Get paired log probabilities
        paired_logprobs = self.get_paired_logprobs_with_reference(
            prompts, chosen_responses, rejected_responses, batch_size
        )
        
        # Calculate rewards for chosen responses using total log probabilities
        chosen_total_rewards = self.calculate_rewards(
            paired_logprobs["policy_chosen_total_logprobs"],
            paired_logprobs["ref_chosen_total_logprobs"]
        )
        
        # Calculate rewards for rejected responses using total log probabilities
        rejected_total_rewards = self.calculate_rewards(
            paired_logprobs["policy_rejected_total_logprobs"],
            paired_logprobs["ref_rejected_total_logprobs"]
        )
        
        # Also calculate average rewards for comparison
        chosen_avg_rewards = self.calculate_rewards(
            paired_logprobs["policy_chosen_avg_logprobs"],
            paired_logprobs["ref_chosen_avg_logprobs"]
        )
        
        rejected_avg_rewards = self.calculate_rewards(
            paired_logprobs["policy_rejected_avg_logprobs"],
            paired_logprobs["ref_rejected_avg_logprobs"]
        )
        
        return {
            "chosen_total_rewards": chosen_total_rewards,
            "rejected_total_rewards": rejected_total_rewards,
            "chosen_avg_rewards": chosen_avg_rewards,
            "rejected_avg_rewards": rejected_avg_rewards,
            # Use total rewards as the default rewards
            "chosen_rewards": chosen_total_rewards,
            "rejected_rewards": rejected_total_rewards,
            "policy_chosen_total_logprobs": paired_logprobs["policy_chosen_total_logprobs"],
            "policy_rejected_total_logprobs": paired_logprobs["policy_rejected_total_logprobs"],
            "ref_chosen_total_logprobs": paired_logprobs["ref_chosen_total_logprobs"],
            "ref_rejected_total_logprobs": paired_logprobs["ref_rejected_total_logprobs"],
            "policy_chosen_avg_logprobs": paired_logprobs["policy_chosen_avg_logprobs"],
            "policy_rejected_avg_logprobs": paired_logprobs["policy_rejected_avg_logprobs"],
            "ref_chosen_avg_logprobs": paired_logprobs["ref_chosen_avg_logprobs"],
            "ref_rejected_avg_logprobs": paired_logprobs["ref_rejected_avg_logprobs"]
        }

    def calculate_dpo_loss(
        self,
        policy_chosen_logprobs: List[float],
        policy_rejected_logprobs: List[float],
        ref_chosen_logprobs: List[float],
        ref_rejected_logprobs: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate the DPO loss and rewards.
        
        DPO loss = -log(sigmoid(beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))))
        """
        # Convert to tensors
        policy_chosen_tensor = torch.tensor(policy_chosen_logprobs)
        policy_rejected_tensor = torch.tensor(policy_rejected_logprobs)
        ref_chosen_tensor = torch.tensor(ref_chosen_logprobs)
        ref_rejected_tensor = torch.tensor(ref_rejected_logprobs)
        
        # Calculate log ratios
        chosen_log_ratios = policy_chosen_tensor - ref_chosen_tensor
        rejected_log_ratios = policy_rejected_tensor - ref_rejected_tensor
        
        # Calculate rewards
        chosen_rewards = self.beta * chosen_log_ratios
        rejected_rewards = self.beta * rejected_log_ratios
        
        # Calculate loss
        logits = chosen_rewards - rejected_rewards
        losses = -torch.nn.functional.logsigmoid(logits)
        
        return losses.tolist(), chosen_rewards.tolist(), rejected_rewards.tolist()
    
    def get_dpo_loss_and_rewards(
        self,
        prompts: Union[str, List[str]],
        chosen_responses: Union[str, List[str]],
        rejected_responses: Union[str, List[str]],
        batch_size: int = 1
    ) -> Dict[str, List[float]]:
        """
        Calculate DPO loss and rewards for chosen and rejected responses.
        """
        # Get paired log probabilities with reference
        paired_logprobs = self.get_paired_logprobs_with_reference(
            prompts, chosen_responses, rejected_responses, batch_size
        )
        
        # Calculate DPO loss and rewards
        dpo_losses, chosen_rewards, rejected_rewards = self.calculate_dpo_loss(
            paired_logprobs["policy_chosen_avg_logprobs"],
            paired_logprobs["policy_rejected_avg_logprobs"],
            paired_logprobs["ref_chosen_avg_logprobs"],
            paired_logprobs["ref_rejected_avg_logprobs"]
        )
        
        return {
            "dpo_losses": dpo_losses,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "policy_chosen_logprobs": paired_logprobs["policy_chosen_avg_logprobs"],
            "policy_rejected_logprobs": paired_logprobs["policy_rejected_avg_logprobs"],
            "ref_chosen_logprobs": paired_logprobs["ref_chosen_avg_logprobs"],
            "ref_rejected_logprobs": paired_logprobs["ref_rejected_avg_logprobs"]
        }

    def get_logprobs_for_class(
        self,
        prompts: Union[str, List[str]],
        class_responses: List[str],
        batch_size: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate log probabilities for multiple class responses per prompt.
        """
        # Convert single prompt to list if needed
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Repeat prompts for each class
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * len(class_responses))
            
        # Calculate log probabilities
        results = self.get_logprobs(
            prompts=expanded_prompts,
            responses=class_responses * len(prompts),
            batch_size=batch_size
        )
        
        # Organize results by prompt and class
        organized_results = {}
        for i, prompt in enumerate(prompts):
            class_logprobs = {}
            start_idx = i * len(class_responses)
            for j, class_response in enumerate(class_responses):
                class_logprobs[class_response] = {
                    "total_logprob": results["total_logprobs"][start_idx + j],
                    "avg_logprob": results["avg_logprobs"][start_idx + j]
                }
            organized_results[prompt] = class_logprobs
            
        return organized_results
    
    def get_class_rewards(
            self,
            prompts: Union[str, List[str]],
            class_responses: List[str],
            batch_size: int = 1
        ) -> Dict[str, Dict[str, Dict[str, float]]]:
            """
            Calculate rewards for multiple class responses per prompt using total log probabilities.
            """
            if self.ref_model is None:
                raise ValueError("Reference model is not provided. Unable to calculate rewards.")
                
            # Convert single prompt to list if needed
            if isinstance(prompts, str):
                prompts = [prompts]
                
            # Repeat prompts for each class
            expanded_prompts = []
            for prompt in prompts:
                expanded_prompts.extend([prompt] * len(class_responses))
                
            # Get log probabilities from both models
            policy_results = self.get_logprobs(
                prompts=expanded_prompts,
                responses=class_responses * len(prompts),
                batch_size=batch_size
            )
            
            # Calculate reference log probabilities
            ref_total_results = []
            ref_avg_results = []
            
            # Process in batches
            for i in range(0, len(expanded_prompts), batch_size):
                batch_prompts = expanded_prompts[i:i + batch_size]
                batch_responses = (class_responses * len(prompts))[i:i + batch_size]
                
                # Prepare inputs and use data collator
                inputs = self.prepare_inputs(batch_prompts, batch_responses)
                batch = self.data_collator(inputs)
                
                # Adjust batch padding
                batch = adjust_batch_padding(batch)
                
                # Move batch to the device
                batch = {k: v.to(self.trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Calculate reference log probabilities
                with torch.no_grad():
                    ref_logps, _, _, _, ref_avg_logps = self.trainer.concatenated_forward(
                        self.ref_model,
                        batch
                    )
                    ref_total_results.extend(ref_logps.cpu().tolist())
                    ref_avg_results.extend(ref_avg_logps.cpu().tolist())
            
            # Calculate rewards using total log probabilities
            total_rewards = []
            for i in range(len(policy_results["total_logprobs"])):
                total_reward = self.beta * (policy_results["total_logprobs"][i] - ref_total_results[i])
                total_rewards.append(total_reward)
            
            # Also calculate average rewards for comparison
            avg_rewards = []
            for i in range(len(policy_results["avg_logprobs"])):
                avg_reward = self.beta * (policy_results["avg_logprobs"][i] - ref_avg_results[i])
                avg_rewards.append(avg_reward)
            
            # Organize results by prompt and class
            organized_results = {}
            for i, prompt in enumerate(prompts):
                class_results = {}
                start_idx = i * len(class_responses)
                for j, class_response in enumerate(class_responses):
                    idx = start_idx + j
                    class_results[class_response] = {
                        "policy_total_logprob": policy_results["total_logprobs"][idx],
                        "policy_avg_logprob": policy_results["avg_logprobs"][idx],
                        "ref_total_logprob": ref_total_results[idx],
                        "ref_avg_logprob": ref_avg_results[idx],
                        "total_reward": total_rewards[idx],
                        "avg_reward": avg_rewards[idx],
                        # Use total_reward as the default reward
                        "reward": total_rewards[idx]
                    }
                organized_results[prompt] = class_results
                
            return organized_results