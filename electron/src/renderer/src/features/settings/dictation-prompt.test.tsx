import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';
const mock = vi.hoisted(() => ({ apiJson: vi.fn() }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@/components/dictation-demo', () => ({ DictationDemo: () => null }));
vi.mock('@/lib/api/client', () => ({ apiJson: mock.apiJson }));
vi.mock('@/hooks/use-native-dictation', () => ({
  dictationPreferencesKey: ['prefs'],
  nativeShortcutKey: ['native-shortcut'],
  useDictationPreferences: () => ({ data: { enabled: true, mode: 'hold', prompt: 'gRPC' } }),
}));
import { ShortcutSettings } from './shortcut-settings';
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
function renderSettings() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ShortcutSettings />
    </QueryClientProvider>,
  );
  return {
    field: screen.getByRole('textbox', { name: 'voicePanel.prompt_label' }),
    save: screen.getByRole('button', { name: 'common.save' }),
  };
}
it('saves an edited vocabulary prompt and stays idle while unchanged', async () => {
  mock.apiJson.mockResolvedValue({});
  const { field, save } = renderSettings();
  expect(field).toHaveValue('gRPC');
  expect(save).toBeDisabled();
  fireEvent.change(field, { target: { value: 'gRPC, Kubernetes' } });
  expect(save).toBeEnabled();
  fireEvent.click(save);
  await waitFor(() => expect(mock.apiJson).toHaveBeenCalledTimes(1));
  const [path, init] = mock.apiJson.mock.calls[0];
  expect(path).toBe('/dictation/prefs');
  expect(JSON.parse(init.body)).toEqual({ prompt: 'gRPC, Kubernetes' });
});
it('reports a failed save next to the prompt', async () => {
  mock.apiJson.mockRejectedValue(new Error('offline'));
  const { field, save } = renderSettings();
  fireEvent.change(field, { target: { value: '' } });
  fireEvent.click(save);
  expect(await screen.findByRole('alert')).toHaveTextContent('common.error');
});
