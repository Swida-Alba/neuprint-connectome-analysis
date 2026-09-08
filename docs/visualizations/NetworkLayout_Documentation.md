# Network Layout Parameter Documentation

## Overview
The `network_layout` parameter in the `FindNeuronConnection` class controls how nodes are arranged in interactive network visualizations.

## Parameter

### `network_layout: str = 'layered'`

**Location**: Class attribute of `FindNeuronConnection`

**Default**: `'layered'`

**Options**:
- `'layered'`: Multipartite layout
- `'distributed'`: Spring layout with layer-based initialization

## Layout Algorithms

### 1. Layered Layout (`'layered'`)

**Algorithm**: NetworkX `multipartite_layout`

**Characteristics**:
- Arranges nodes in distinct horizontal layers
- Nodes are positioned strictly according to their layer number
- All nodes in the same layer are aligned vertically
- Preserves the hierarchical structure clearly

**Best for**:
- ✅ Strictly hierarchical networks
- ✅ Feed-forward networks with no cross-layer connections
- ✅ Visualizing layer-by-layer progression
- ✅ Networks where temporal/spatial ordering is important

**Example Use Cases**:
- Standard shortest path analysis
- Sequential processing pathways
- Clear hierarchical relationships

**Visualization**:
```
Layer 0    Layer 1    Layer 2    Layer 3
  O           O           O          O
  |           |          / \         |
  O    →→→    O    →→→  O   O   →→→  O
  |           |          \ /         |
  O           O           O          O
```

### 2. Distributed Layout (`'distributed'`)

**Algorithm**: NetworkX `spring_layout` with multipartite initialization

**Characteristics**:
- Uses force-directed algorithm
- Starts with layer-based positions (from multipartite)
- Spreads nodes apart for better clarity
- Optimizes node positions to minimize edge crossings
- May slightly alter layer structure for clarity

**Parameters**:
- `k=1.5`: Optimal distance between nodes
- `iterations=50`: Number of optimization iterations
- `seed=42`: Fixed seed for reproducible layouts

**Best for**:
- ✅ Networks with cross-layer connections
- ✅ Complex connectivity patterns
- ✅ Dense networks where layered layout causes overlaps
- ✅ Networks with feedback loops or recurrent connections

**Example Use Cases**:
- Networks with lateral connections
- Recurrent neural networks
- Complex multi-pathway systems
- Networks where strict layering hides important connections

**Visualization**:
```
         O ←─┐
        / \  │
       O   O │
      /│\ /│\│
     O O O O O
      \│/ \│/
       O   O
        \ /
         O
```

## Usage

### Setting the Parameter

```python
from coana import FindNeuronConnection

# Use layered layout (default)
fnc = FindNeuronConnection(
    sourceNeurons=['PPL1'],
    targetNeurons=['MBON.*'],
    network_layout='layered',
    token='your_token_here'
)

# Use distributed layout for complex networks
fnc = FindNeuronConnection(
    sourceNeurons=['PPL1'],
    targetNeurons=['MBON.*'],
    network_layout='distributed',
    token='your_token_here'
)
```

### When to Change from Default

**Keep `'layered'` (default) when**:
- Your network is strictly feed-forward
- You want to emphasize the layer structure
- All connections go from layer N to layer N+1
- The visualization is clear and readable

**Change to `'distributed'` when**:
- You see overlapping nodes in layered layout
- Connections cross multiple layers
- There are feedback or recurrent connections
- The network has complex topology
- You need to identify clusters or communities

## Implementation Details

### Layout Generation Method

```python
def _get_network_layout(self, G):
    '''Get network layout based on network_layout parameter'''
    if self.network_layout == 'layered':
        # Multipartite layout - nodes arranged in layers
        pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
    elif self.network_layout == 'distributed':
        # Spring layout with layer-based initial positions
        initial_pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
        pos = nx.spring_layout(G, pos=initial_pos, k=1.5, iterations=50, seed=42)
    else:
        raise ValueError(f"network_layout must be 'layered' or 'distributed'")
    return pos
```

### Where It's Applied

The `network_layout` parameter affects both:
1. **Type-based networks**: Nodes represent neuron types
2. **BodyId-based networks**: Nodes represent individual neurons

Applied in both methods:
- `FindPath()`: Shortest paths visualization
- `FindAllPath()`: All paths visualization

## Output Files

Both layout options generate the same output files with different node positions:

**FindPath()**:
- `visualization/Network_<run>.html`
- `bodyId_visualization/Network_<run>.html`

**FindAllPath()**:
- `visualization/Network_<run>.html`
- `bodyId_visualization/Network_<run>.html`

The run-folder name carries the parameters and timestamp. Type-level and
bodyId-level artifacts intentionally share the same filename format; the
containing folder identifies the level. Early previews are written under
`network_early/` and `network_early_bodyId/`.

## Comparison Example

### Same Network, Different Layouts

**Layered Layout**:
- Pros: Clear layer progression, easy to identify layer boundaries
- Cons: May have node overlaps in dense layers, cross-layer edges look messy

**Distributed Layout**:
- Pros: Better node spacing, clearer complex connections, reveals network structure
- Cons: Layer boundaries less obvious, may seem less organized

## Performance Considerations

### Layered Layout
- **Speed**: Very fast (O(n) time)
- **Memory**: Minimal
- **Deterministic**: Always produces same layout

### Distributed Layout
- **Speed**: Slower (O(n² * iterations) time)
- **Memory**: Higher (stores forces and velocities)
- **Deterministic**: Yes (due to fixed seed)
- **Note**: 50 iterations is optimized for balance between quality and speed

## Tips and Best Practices

1. **Start with default (`'layered'`)**:
   - Most networks benefit from the clear hierarchy
   - Easier to understand layer-by-layer flow

2. **Switch to `'distributed'` if**:
   - Nodes are overlapping
   - Many edges cross multiple layers
   - Network structure isn't purely hierarchical

3. **Experiment**:
   - Try both layouts to see which reveals your network better
   - Different networks may benefit from different layouts

4. **Consider network size**:
   - Layered works better for large networks (faster)
   - Distributed works better for medium-sized complex networks

5. **Check the results**:
   - Open both HTML files
   - Compare readability
   - Choose the one that best shows your data

## Error Handling

If an invalid value is provided:
```python
ValueError: network_layout must be 'layered' or 'distributed', got 'invalid'
```

Valid values are case-sensitive and must be exactly:
- `'layered'`
- `'distributed'`

## Future Enhancements

Potential additional layouts that could be added:
- `'circular'`: Arrange nodes in concentric circles
- `'hierarchical'`: Tree-like layout for diverging/converging patterns
- `'force_atlas'`: More sophisticated force-directed layout
- `'kamada_kawai'`: Energy-based layout for small networks

## References

- NetworkX multipartite_layout: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.multipartite_layout.html
- NetworkX spring_layout: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
