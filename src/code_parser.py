import tree_sitter_languages as tsl
from pathlib import Path
import hashlib
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CSharpParser:
    def __init__(self):
        self.parser = tsl.get_parser("c_sharp")
        
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a C# file and extract structured information"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = self.parser.parse(content.encode())
            
            # Extract metadata
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'content': content,
                'hash': hashlib.md5(content.encode()).hexdigest(),
                'classes': [],
                'methods': [],
                'interfaces': [],
                'namespaces': [],
                'usings': []
            }
            
            # Parse tree to extract structures
            self._extract_structures(tree.root_node, content, metadata)
            
            # Create document with chunks
            return self._create_document(metadata)
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_structures(self, node, source_code: str, metadata: Dict):
        """Recursively extract code structures"""
        if node.type == 'namespace_declaration':
            namespace = source_code[node.start_byte:node.end_byte]
            metadata['namespaces'].append({
                'name': self._get_namespace_name(node, source_code),
                'content': namespace,
                'start_line': node.start_point[0],
                'end_line': node.end_point[0]
            })
        
        elif node.type == 'class_declaration':
            class_name = self._get_class_name(node, source_code)
            metadata['classes'].append({
                'name': class_name,
                'content': source_code[node.start_byte:node.end_byte],
                'start_line': node.start_point[0],
                'end_line': node.end_point[0],
                'methods': self._extract_methods(node, source_code)
            })
        
        elif node.type == 'interface_declaration':
            interface_name = self._get_interface_name(node, source_code)
            metadata['interfaces'].append({
                'name': interface_name,
                'content': source_code[node.start_byte:node.end_byte],
                'start_line': node.start_point[0],
                'end_line': node.end_point[0]
            })
        
        elif node.type == 'using_directive':
            using = source_code[node.start_byte:node.end_byte]
            metadata['usings'].append(using)
        
        # Recurse through children
        for child in node.children:
            self._extract_structures(child, source_code, metadata)
    
    def _extract_methods(self, class_node, source_code: str) -> List[Dict]:
        """Extract methods from a class node"""
        methods = []
        for child in class_node.children:
            if child.type == 'method_declaration':
                method_name = self._get_method_name(child, source_code)
                methods.append({
                    'name': method_name,
                    'signature': self._get_method_signature(child, source_code),
                    'content': source_code[child.start_byte:child.end_byte],
                    'start_line': child.start_point[0],
                    'end_line': child.end_point[0]
                })
        return methods
    
    def _get_namespace_name(self, node, source_code: str) -> str:
        """Extract namespace name from node"""
        for child in node.children:
            if child.type == 'identifier' or child.type == 'qualified_name':
                return source_code[child.start_byte:child.end_byte]
        return "Unknown"
    
    def _get_class_name(self, node, source_code: str) -> str:
        """Extract class name from node"""
        for child in node.children:
            if child.type == 'identifier':
                return source_code[child.start_byte:child.end_byte]
        return "Unknown"
    
    def _get_interface_name(self, node, source_code: str) -> str:
        """Extract interface name from node"""
        for child in node.children:
            if child.type == 'identifier':
                return source_code[child.start_byte:child.end_byte]
        return "Unknown"
    
    def _get_method_name(self, node, source_code: str) -> str:
        """Extract method name from node"""
        for child in node.children:
            if child.type == 'identifier':
                return source_code[child.start_byte:child.end_byte]
        return "Unknown"
    
    def _get_method_signature(self, node, source_code: str) -> str:
        """Extract method signature"""
        # Find the parameter list
        for child in node.children:
            if child.type == 'parameter_list':
                end_byte = child.end_byte
                return source_code[node.start_byte:end_byte]
        return source_code[node.start_byte:node.start_byte + 100]  # Fallback
    
    def _create_document(self, metadata: Dict) -> Dict:
        """Create document with intelligent chunks"""
        chunks = []
        
        # Create file-level summary chunk
        summary = f"File: {metadata['file_name']}\n"
        summary += f"Namespaces: {', '.join([ns['name'] for ns in metadata['namespaces']])}\n"
        summary += f"Classes: {', '.join([cls['name'] for cls in metadata['classes']])}\n"
        summary += f"Interfaces: {', '.join([intf['name'] for intf in metadata['interfaces']])}\n"
        
        chunks.append({
            'type': 'file_summary',
            'content': summary,
            'metadata': {
                'file_path': metadata['file_path'],
                'chunk_type': 'summary'
            }
        })
        
        # Create chunks for each class
        for cls in metadata['classes']:
            # Class-level chunk
            class_chunk = f"Class: {cls['name']}\n"
            class_chunk += f"File: {metadata['file_name']}\n"
            class_chunk += f"Methods: {', '.join([m['name'] for m in cls['methods']])}\n\n"
            class_chunk += cls['content'][:1000]  # First 1000 chars
            
            chunks.append({
                'type': 'class',
                'content': class_chunk,
                'metadata': {
                    'file_path': metadata['file_path'],
                    'class_name': cls['name'],
                    'chunk_type': 'class'
                }
            })
            
            # Method-level chunks for large methods
            for method in cls['methods']:
                if len(method['content']) > 500:
                    method_chunk = f"Method: {method['name']}\n"
                    method_chunk += f"Class: {cls['name']}\n"
                    method_chunk += f"File: {metadata['file_name']}\n\n"
                    method_chunk += method['content']
                    
                    chunks.append({
                        'type': 'method',
                        'content': method_chunk,
                        'metadata': {
                            'file_path': metadata['file_path'],
                            'class_name': cls['name'],
                            'method_name': method['name'],
                            'chunk_type': 'method'
                        }
                    })
        
        return {
            'file_path': metadata['file_path'],
            'chunks': chunks,
            'metadata': metadata
        } 