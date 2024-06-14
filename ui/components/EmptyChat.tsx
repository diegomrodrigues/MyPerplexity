import EmptyChatMessageInput from './EmptyChatMessageInput';
import ThemeSwitcher from './theme/Switcher';

const EmptyChat = ({
  sendMessage,
  focusMode,
  setFocusMode,
  copilotEnabled,
  setCopilotEnabled
}: {
  sendMessage: (message: string) => void;
  focusMode: string;
  setFocusMode: (mode: string) => void;
  copilotEnabled: boolean;
  setCopilotEnabled: (copilotEnabled: boolean) => void;
}) => {
  return (
    <div className="relative">
      <ThemeSwitcher className="absolute top-2 right-0 lg:hidden" />

      <div className="flex flex-col items-center justify-center min-h-screen max-w-screen-sm mx-auto p-2 space-y-8">
        <h2 className="text-black/70 dark:text-white/70 text-3xl font-medium -mt-8">
          Perplexity do Diego 😄
        </h2>
        <EmptyChatMessageInput
          sendMessage={sendMessage}
          focusMode={focusMode}
          setFocusMode={setFocusMode}
          copilotEnabled={copilotEnabled}
          setCopilotEnabled={setCopilotEnabled}
        />
      </div>
    </div>
  );
};

export default EmptyChat;
