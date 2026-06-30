import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Shown in place of the crashed subtree; defaults to a generic message. */
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches render-time exceptions in a subtree (e.g. a malformed SSE/API
 * payload reaching `applySongEvent` or a card's render) so one bad song/row
 * shows an inline error instead of blanking the whole list. React error
 * boundaries are class-only — there's no hook equivalent.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div
            role="alert"
            className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 p-4 text-sm text-red-700 dark:text-red-300"
          >
            Something went wrong rendering this section. Try reloading the page.
          </div>
        )
      );
    }
    return this.props.children;
  }
}
