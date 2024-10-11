import React, { useState, useRef, useEffect } from 'react';
import Editor from "@monaco-editor/react";
import { Container, Typography, Select, MenuItem, Box, Button } from '@mui/material';

const languages = [
  { value: 'python', label: 'Python' },
  { value: 'markdown', label: 'Markdown' },
  { value: 'javascript', label: 'JavaScript' }
];

const initialCode = {
  python: '# Python code here\nprint("Hello, World!")',
  markdown: '# Markdown content here\n\nThis is a **bold** text.',
  javascript: '// JavaScript code here\nconsole.log("Hello, World!");'
};

export default function App() {
  const [language, setLanguage] = useState('python');
  const [code, setCode] = useState(initialCode.python);
  const editorRef = useRef(null);

  const handleLanguageChange = (event) => {
    const newLanguage = event.target.value;
    setLanguage(newLanguage);
    setCode(initialCode[newLanguage]);
  };

  function handleEditorDidMount(editor, monaco) {
    editorRef.current = editor;
  }

  function saveCode() {
    if (editorRef.current) {
      setCode(editorRef.current.getValue());
    }
  }

  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.setValue(code);
    }
  }, [language]);

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Monaco Editor Demo
      </Typography>
      <Typography variant="body1" gutterBottom>
        Select a language and start coding!
      </Typography>
      
      <Select
        value={language}
        onChange={handleLanguageChange}
        sx={{ mb: 2 }}
      >
        {languages.map((lang) => (
          <MenuItem key={lang.value} value={lang.value}>
            {lang.label}
          </MenuItem>
        ))}
      </Select>
      
      <Box sx={{ border: 1, borderColor: 'grey.300', borderRadius: 1, overflow: 'hidden', mb: 2 }}>
        <Editor
          height="400px"
          language={language}
          value={code}
          theme="vs-dark"
          onMount={handleEditorDidMount}
          options={{
            minimap: { enabled: false }
          }}
        />
      </Box>
      <Button variant="contained" onClick={saveCode} sx={{ mb: 2 }}>
        Save Code
      </Button>
      <Typography variant="h6" component="h2" gutterBottom>
        Saved Code
      </Typography>
      <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>{code}</pre>
    </Container>
  );
}