from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm_bar
import jieba

from deepeval.models import DeepEvalBaseLLM

from ... import BaseAttack
from .template import (
    StegoTemplate,
)
from .schema import (
    SteganographicParagraph
)
from ...attack_simulator.utils import (
    generate_schema,
    a_generate_schema,
)
from ....utils import judge_language

class Stego(BaseAttack):

    def __init__(self, weight: int = 1, max_retries: int = 3, 
                 generation_template_zh: str = None, decoding_template_zh: str = None,
                 generation_template_en: str = None, decoding_template_en: str = None):
        self.weight = weight
        self.max_retries = max_retries
        self.generation_template_zh = generation_template_zh
        self.decoding_template_zh = decoding_template_zh
        self.generation_template_en = generation_template_en
        self.decoding_template_en = decoding_template_en

    def enhance(self, attack: str, simulator_model: DeepEvalBaseLLM) -> str:
        self.simulator_model = simulator_model
        
        is_chinese = (judge_language(attack) == "chinese")

        if is_chinese:
            words = list(jieba.cut(attack))
            # Choose generator
            if self.generation_template_zh:
                def generate_steganographic_text(word_list):
                    return self.generation_template_zh.replace("{word_list}", str(word_list))
            else:
                generate_steganographic_text = StegoTemplate.generate_steganographic_text_zh
            
            # Choose decoder
            if self.decoding_template_zh:
                def implement_jailbreak(stego_paragraph):
                    return self.decoding_template_zh.replace("{stego_paragraph}", stego_paragraph)
            else:
                implement_jailbreak = StegoTemplate.implement_jailbreak_zh
        else:
            words = attack.split()
            # Choose generator
            if self.generation_template_en:
                def generate_steganographic_text(word_list):
                    return self.generation_template_en.replace("{word_list}", str(word_list))
            else:
                generate_steganographic_text = StegoTemplate.generate_steganographic_text_en
            
            # Choose decoder
            if self.decoding_template_en:
                def implement_jailbreak(stego_paragraph):
                    return self.decoding_template_en.replace("{stego_paragraph}", stego_paragraph)
            else:
                implement_jailbreak = StegoTemplate.implement_jailbreak_en
            
        words = [word for word in words if len(word.strip()) > 0 and word not in "，。？！"""]
        prompt = generate_steganographic_text(word_list=words)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        with tqdm(
            total=self.max_retries * 2,
            desc="...... 🔎 Generating steganographic text",
            unit="step",
            leave=False,
        ) as pbar:

            for _ in range(self.max_retries):
                try:
                    # Generate the enhanced attack
                    res = self._generate_schema(
                        prompt, SteganographicParagraph
                    )
                    
                    # Check if res is a string or an object with paragraph attribute
                    if isinstance(res, str):
                        try:
                            import json
                            # Try to parse JSON if it looks like one
                            if res.strip().startswith('{') and res.strip().endswith('}'):
                                 data = json.loads(res)
                                 stego_paragraph = data.get("paragraph", res)
                            else:
                                 stego_paragraph = res
                        except Exception:
                            stego_paragraph = res
                    elif hasattr(res, 'paragraph'):
                        stego_paragraph = res.paragraph
                    else:
                        stego_paragraph = str(res)
                    pbar.update(1)

                    enhanced_attack = implement_jailbreak(stego_paragraph=stego_paragraph)
                    pbar.update(1)  # Update the progress bar for compliance

                    return enhanced_attack
                except Exception as e:
                    print(f"[Stego Error] Retry failed: {e}")
                    continue

        # If all retries fail, return the original attack
        return attack

    async def a_enhance(
        self, attack: str, simulator_model: DeepEvalBaseLLM
    ) -> str:
        self.simulator_model = simulator_model
        
        is_chinese = (judge_language(attack) == "chinese")

        if is_chinese:
            words = list(jieba.cut(attack))
            if self.generation_template_zh:
                def generate_steganographic_text(word_list):
                    return self.generation_template_zh.replace("{word_list}", str(word_list))
            else:
                generate_steganographic_text = StegoTemplate.generate_steganographic_text_zh
            
            if self.decoding_template_zh:
                def implement_jailbreak(stego_paragraph):
                    return self.decoding_template_zh.replace("{stego_paragraph}", stego_paragraph)
            else:
                implement_jailbreak = StegoTemplate.implement_jailbreak_zh
        else:
            words = attack.split()
            if self.generation_template_en:
                def generate_steganographic_text(word_list):
                    return self.generation_template_en.replace("{word_list}", str(word_list))
            else:
                generate_steganographic_text = StegoTemplate.generate_steganographic_text_en
            
            if self.decoding_template_en:
                def implement_jailbreak(stego_paragraph):
                    return self.decoding_template_en.replace("{stego_paragraph}", stego_paragraph)
            else:
                implement_jailbreak = StegoTemplate.implement_jailbreak_en
            
        words = [word for word in words if len(word.strip()) > 0 and word not in "，。？！"""]
        prompt = generate_steganographic_text(word_list=words)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        pbar = async_tqdm_bar(
            total=self.max_retries * 2,
            desc="...... 🔎 Generating steganographic text",
            unit="step",
            leave=False,
        )

        for _ in range(self.max_retries):
            try:
                res = await self._a_generate_schema(
                    prompt, SteganographicParagraph
                )
                # Check if res is a string or an object with paragraph attribute
                if isinstance(res, str):
                    try:
                        import json
                        # Try to parse JSON if it looks like one
                        if res.strip().startswith('{') and res.strip().endswith('}'):
                             data = json.loads(res)
                             stego_paragraph = data.get("paragraph", res)
                        else:
                             stego_paragraph = res
                    except Exception:
                        stego_paragraph = res
                elif hasattr(res, 'paragraph'):
                    stego_paragraph = res.paragraph
                else:
                    stego_paragraph = str(res)
                pbar.update(1)  # Update the progress bar for generation

                enhanced_attack = implement_jailbreak(stego_paragraph=stego_paragraph)
                pbar.update(1)  # Update the progress bar for compliance
                pbar.close()
                return enhanced_attack
            except Exception as e:
                print(f"[Stego Async Error] Retry failed: {e}")
                continue

        # Close the progress bar after the loop
        pbar.close()

        # If all retries fail, return the original attack
        return attack

    ##################################################
    ### Helper Methods ################################
    ##################################################

    def _generate_schema(self, prompt: str, schema: BaseModel):
        return generate_schema(prompt, schema, self.simulator_model)

    async def _a_generate_schema(self, prompt: str, schema: BaseModel):
        return await a_generate_schema(prompt, schema, self.simulator_model)

    def get_name(self) -> str:
        return "Stego"
