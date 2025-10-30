# Mermaid Diagrams for Presentation

This folder contains Mermaid diagram definitions used in the presentation slides.

## Diagrams

1. **dask-workflow.mmd** - Shows how Dask partitions data and distributes work across workers
2. **memory-aware-pipeline.mmd** - Illustrates the 5-step memory-aware chunking pipeline
3. **algorithm-flowchart.mmd** - Detailed flowchart of the chunk size computation algorithm
4. **comparison-architecture.mmd** - Contrasts traditional vs memory-aware chunking approaches

## Viewing Diagrams

### Online Editors

1. **Mermaid Live Editor**: https://mermaid.live/
   - Copy and paste diagram code
   - Export as PNG or SVG
   - Edit interactively

2. **GitHub/GitLab**:
   - GitHub and GitLab natively render `.mmd` files
   - Just view the file in the repository

### VS Code Extension

Install the "Mermaid Preview" extension:
```bash
code --install-extension bierner.markdown-mermaid
```

Then open any `.mmd` file and press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows/Linux) to preview.

### Command Line

Using the Mermaid CLI:
```bash
# Install
npm install -g @mermaid-js/mermaid-cli

# Generate PNG
mmdc -i dask-workflow.mmd -o dask-workflow.png

# Generate SVG
mmdc -i dask-workflow.mmd -o dask-workflow.svg -t default
```

## Editing Diagrams

Mermaid syntax is straightforward:

- `graph LR` = Left-to-right flowchart
- `graph TD` = Top-down flowchart
- `A[Text]` = Node with text
- `A --> B` = Arrow from A to B
- `A -->|Label| B` = Labeled arrow
- `style A fill:#color` = Node styling

See the [Mermaid documentation](https://mermaid.js.org/) for full syntax.

## Integration with Slides

These diagrams are embedded directly in `slides.md` using code blocks:

\`\`\`mermaid
graph LR
    A --> B
\`\`\`

The Reveal.js presentation automatically renders them using the Mermaid JavaScript library.

## Customization

To change diagram appearance, edit the `themeVariables` in `index.html`:

```javascript
mermaid.initialize({ 
    theme: 'default',
    themeVariables: {
        primaryColor: '#e1f5ff',
        primaryBorderColor: '#2a5a7a',
        // ... other colors
    }
});
```

