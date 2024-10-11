import React, { useState } from 'react';
import Editor from "@monaco-editor/react";
import { Container, Typography, Select, MenuItem, Box } from '@mui/material';

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

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

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
      
      <Box sx={{ border: 1, borderColor: 'grey.300', borderRadius: 1, overflow: 'hidden' }}>
        <Editor
          height="400px"
          language={language}
          value={initialCode[language]}
          theme="vs-dark"
          options={{
            minimap: { enabled: false }
          }}
        />
      </Box>
    </Container>
  );
}