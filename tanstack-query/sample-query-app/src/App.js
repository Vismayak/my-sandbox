import './App.css';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Posts from './posts';
import CreatePost from './CreatePost';

const queryClient = new QueryClient();

function App() {

  return (
    <div className="App">
      <QueryClientProvider client={queryClient}>
        <CreatePost />
        <Posts />
      </QueryClientProvider>
    </div>
  );
}

export default App;
