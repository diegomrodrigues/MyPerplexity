import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { ChatResult } from "@langchain/core/outputs";
import {
  BaseChatModel,
  BaseChatModelCallOptions,
  BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

export interface DummyChatModelOptions
  extends BaseChatModelCallOptions {}

export interface DummyChatModelParams extends BaseChatModelParams {
}

export class DummyChatModel extends BaseChatModel<DummyChatModelOptions> {

  static lc_name(): string {
    return "DummyChatModel";
  }

  constructor(fields: DummyChatModelParams) {
    super(fields);
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    if (!messages.length) {
      throw new Error("No messages provided.");
    }
    if (typeof messages[0].content !== "string") {
      throw new Error("Multimodal messages are not supported.");
    }
    // Pass `runManager?.getChild()` when invoking internal runnables to enable tracing
    // await subRunnable.invoke(params, runManager?.getChild());
    const content = messages[0].content;
    const tokenUsage = {};
    return {
      generations: [{ message: new AIMessage({ content }), text: content }],
      llmOutput: { content },
    };
  }

  _llmType(): string {
    return "dummy_chat_model";
  }
}