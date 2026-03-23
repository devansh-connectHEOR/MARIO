"""
scripts/image_llm.py

Provides ImageLLM, a thin subclass of ChatOpenAI that works around OpenAI's
restriction on image content inside ToolMessages.

Problem:
    When a LangChain tool returns image data (e.g. extracted PDF images), the
    result is placed in a ToolMessage. OpenAI's API rejects image_url blocks
    in ToolMessages, causing a validation error.

Solution:
    Override the internal `_generate` / `_agenerate` methods to intercept the
    message list before it reaches the OpenAI API. Images are stripped from
    any ToolMessage and re-injected as a follow-up HumanMessage, which OpenAI
    accepts without issue. Text content in the original ToolMessage is preserved.

Usage:
    Drop-in replacement for ChatOpenAI:

        from scripts.image_llm import ImageLLM
        llm = ImageLLM(model="gpt-4o")

Note on commented-out methods:
    invoke / ainvoke / stream / astream overrides were tested but are not
    needed — LangChain routes all calls through _generate / _agenerate at
    the base level, so overriding those two is sufficient to intercept all
    usage patterns (sync, async, streaming, and non-streaming).
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from typing import Any, Optional


class ImageLLM(ChatOpenAI):
    """
    A ChatOpenAI subclass that moves image content from ToolMessages to
    HumanMessages before the message list is sent to the OpenAI API.

    OpenAI does not support `image_url` blocks in ToolMessages. This class
    intercepts the message list at the `_generate` / `_agenerate` level,
    splits any ToolMessage with mixed text+image content into:
        - A text-only ToolMessage  (kept in place)
        - A HumanMessage           (appended immediately after, carrying the images)
    """

    def _convert_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Rewrite the message list so that no ToolMessage contains image content.

        For each ToolMessage whose content is a list of blocks:
            - Text blocks are kept in the ToolMessage.
            - Image blocks are extracted and placed in a new HumanMessage
              inserted immediately after the ToolMessage.

        All other message types are passed through unchanged.

        Args:
            messages (list[BaseMessage]): The original message list from LangChain.

        Returns:
            list[BaseMessage]: A rewritten message list safe to send to OpenAI.
        """
        fixed_messages = []

        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                text_blocks = [b for b in msg.content if b.get("type") == "text"] # type: ignore
                image_blocks = [b for b in msg.content if b.get("type") == "image_url"] # type: ignore

                # Keep the ToolMessage, but strip it down to text only
                msg.content = text_blocks if text_blocks else "Image retrieved."
                fixed_messages.append(msg)

                # Inject a HumanMessage to carry the images forward
                if image_blocks:
                    fixed_messages.append(HumanMessage(content=image_blocks))
            else:
                fixed_messages.append(msg)

        return fixed_messages

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Sync generation — intercepts messages before forwarding to OpenAI.

        Args:
            messages: Message list from LangChain.
            stop: Optional list of stop sequences.
            run_manager: Optional LangChain callback manager.
            **kwargs: Additional keyword arguments passed to the parent.

        Returns:
            ChatResult from the parent ChatOpenAI._generate.
        """
        messages = self._convert_messages(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Async generation — intercepts messages before forwarding to OpenAI.

        Args:
            messages: Message list from LangChain.
            stop: Optional list of stop sequences.
            run_manager: Optional LangChain async callback manager.
            **kwargs: Additional keyword arguments passed to the parent.

        Returns:
            ChatResult from the parent ChatOpenAI._agenerate.
        """
        messages = self._convert_messages(messages)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    # --- Commented-out overrides ---
    #
    # invoke, ainvoke, stream, and astream were tested as override points but
    # are unnecessary: LangChain ultimately routes all of these through
    # _generate / _agenerate, so the two overrides above cover all call paths.
    # Left here for reference in case the LangChain internals change.
    #
    # def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
    #     if isinstance(input, list):
    #         input = self._convert_messages(input)
    #     return super().invoke(input, config, **kwargs)
    #
    # async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
    #     if isinstance(input, list):
    #         input = self._convert_messages(input)
    #     return await super().ainvoke(input, config, **kwargs)
    #
    # def stream(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
    #     if isinstance(input, list):
    #         input = self._convert_messages(input)
    #     return super().stream(input, config, **kwargs)
    #
    # async def astream(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
    #     if isinstance(input, list):
    #         input = self._convert_messages(input)
    #     async for chunk in super().astream(input, config, **kwargs):
    #         yield chunk