"""Host-only Chat Completions translation for the actual Responses-native CLI.

No credential handling lives here. Native function calls and their real outputs
are translated both ways; synthetic protocol events never imply synthetic work.
"""
import json
import time
import uuid


def chat_request(body):
    messages = []
    if body.get('instructions'):
        messages.append({'role': 'system', 'content': body['instructions']})
    pending_reasoning = None
    omitted = []
    for item in body.get('input', []):
        kind = item.get('type', 'message')
        if kind == 'reasoning':
            # The bridge carries the provider's opaque encrypted thinking block
            # unchanged. Public summaries are not relabelled encrypted content.
            pending_reasoning = item.get('encrypted_content')
        elif kind in ('function_call', 'custom_tool_call'):
            name = item['name']
            arguments = item.get('arguments')
            if kind == 'custom_tool_call':
                arguments = json.dumps({'input': item.get('input', '')})
            call = {'id': item['call_id'], 'type': 'function',
                    'function': {'name': name, 'arguments': arguments or '{}'}}
            if messages and messages[-1]['role'] == 'assistant' and messages[-1].get('tool_calls'):
                messages[-1]['tool_calls'].append(call)
            else:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
                if pending_reasoning:
                    message['encrypted_content'] = pending_reasoning
                messages.append(message)
            pending_reasoning = None
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            output = item.get('output', '')
            if isinstance(output, list):
                output = '\n'.join(part.get('text', '') for part in output if isinstance(part, dict))
            elif not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': output})
        elif kind == 'message':
            content = item.get('content', '')
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get('type') in ('input_text', 'output_text', 'text'):
                        parts.append({'type': 'text', 'text': part.get('text', '')})
                    elif part.get('type') == 'input_image':
                        parts.append({'type': 'image_url', 'image_url': {'url': part['image_url']}})
                    else:
                        raise ValueError('Unsupported native message content type')
                content = parts
            role = item.get('role', 'user')
            messages.append({'role': 'system' if role == 'developer' else role, 'content': content})
        else:
            raise ValueError('Unsupported native input item: ' + str(kind))
    tools = []
    custom = set()
    for tool in body.get('tools', []):
        if tool.get('type') == 'function':
            function = {k: tool[k] for k in ('name', 'description', 'parameters') if k in tool}
            tools.append({'type': 'function', 'function': function})
        elif tool.get('type') == 'custom':
            custom.add(tool['name'])
            tools.append({'type': 'function', 'function': {
                'name': tool['name'], 'description': tool.get('description', ''),
                'parameters': {'type': 'object', 'properties': {'input': {'type': 'string'}},
                               'required': ['input'], 'additionalProperties': False}}})
        else:
            # Responses-only namespaces / builtin web search are unavailable on
            # Chat Completions. Native shell and file tools remain real CLI tools.
            omitted.append(tool.get('type'))
    request = {'model': body['model'], 'messages': messages, 'stream': True,
               'stream_options': {'include_usage': True},
               'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
    if tools:
        request['tools'] = tools
        choice = body.get('tool_choice', 'auto')
        if isinstance(choice, dict) and choice.get('type') == 'function':
            choice = {'type': 'function', 'function': {'name': choice['name']}}
        request['tool_choice'] = choice
    for field in ('temperature', 'top_p', 'parallel_tool_calls'):
        if field in body:
            request[field] = body[field]
    if body.get('max_output_tokens'):
        request['max_tokens'] = body['max_output_tokens']
    return request, custom, omitted


class ResponseEvents:
    def __init__(self, model, custom_tools=()):
        self.model = model
        self.custom_tools = set(custom_tools)
        self.response_id = 'resp_chat_' + uuid.uuid4().hex
        self.created = int(time.time())
        self.items = []
        self.calls = {}
        self.message = None
        self.reasoning = None
        self.encrypted = ''
        self.usage = None
        self.provider_models = set()
        self.provider_id = None
        self.finish_reason = None
        self.saw_done = False

    def response(self, status):
        value = {'id': self.response_id, 'object': 'response', 'created_at': self.created,
                 'status': status, 'model': self.model, 'output': self.items}
        if self.usage:
            value['usage'] = self.usage
        return value

    def begin(self):
        return [{'type': 'response.created', 'response': self.response('in_progress')},
                {'type': 'response.in_progress', 'response': self.response('in_progress')}]

    def add(self, item):
        index = len(self.items)
        self.items.append(item)
        return index, {'type': 'response.output_item.added', 'output_index': index, 'item': dict(item)}

    def chunk(self, data):
        events = []
        if data.get('model'):
            self.provider_models.add(data['model'])
        self.provider_id = data.get('id', self.provider_id)
        usage = data.get('usage')
        if usage:
            self.usage = {'input_tokens': usage.get('prompt_tokens', 0),
                          'output_tokens': usage.get('completion_tokens', 0),
                          'total_tokens': usage.get('total_tokens', 0),
                          'output_tokens_details': {'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)}}
        for choice in data.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Only a single model choice is supported')
            delta = choice.get('delta', {})
            text = delta.get('reasoning_content')
            if text:
                if self.reasoning is None:
                    item = {'type': 'reasoning', 'id': 'rs_chat_' + uuid.uuid4().hex,
                            'summary': [{'type': 'summary_text', 'text': ''}]}
                    index, event = self.add(item)
                    self.reasoning = (index, item)
                    events.extend([event, {'type': 'response.reasoning_summary_part.added',
                                          'item_id': item['id'], 'output_index': index,
                                          'summary_index': 0, 'part': dict(item['summary'][0])}])
                index, item = self.reasoning
                item['summary'][0]['text'] += text
                events.append({'type': 'response.reasoning_summary_text.delta', 'item_id': item['id'],
                               'output_index': index, 'summary_index': 0, 'delta': text})
            encrypted = delta.get('encrypted_content')
            if encrypted:
                self.encrypted += encrypted
            text = delta.get('content')
            if text:
                if self.message is None:
                    item = {'type': 'message', 'id': 'msg_chat_' + uuid.uuid4().hex,
                            'role': 'assistant', 'status': 'in_progress',
                            'content': [{'type': 'output_text', 'text': '', 'annotations': []}]}
                    index, event = self.add(item)
                    self.message = (index, item)
                    events.extend([event, {'type': 'response.content_part.added', 'item_id': item['id'],
                                          'output_index': index, 'content_index': 0,
                                          'part': dict(item['content'][0])}])
                index, item = self.message
                item['content'][0]['text'] += text
                events.append({'type': 'response.output_text.delta', 'item_id': item['id'],
                               'output_index': index, 'content_index': 0, 'delta': text})
            for call in delta.get('tool_calls', []):
                number = call['index']
                if number not in self.calls:
                    fn = call.get('function', {})
                    item = {'type': 'function_call', 'id': 'fc_chat_' + uuid.uuid4().hex,
                            'call_id': call.get('id') or 'call_chat_' + uuid.uuid4().hex,
                            'name': fn.get('name', ''), 'arguments': '', 'status': 'in_progress'}
                    index, event = self.add(item)
                    self.calls[number] = (index, item)
                    events.append(event)
                index, item = self.calls[number]
                fn = call.get('function', {})
                if call.get('id'):
                    item['call_id'] = call['id']
                if fn.get('name'):
                    item['name'] = fn['name']
                arguments = fn.get('arguments', '')
                if arguments:
                    item['arguments'] += arguments
                    events.append({'type': 'response.function_call_arguments.delta',
                                   'item_id': item['id'], 'output_index': index, 'delta': arguments})
            if choice.get('finish_reason'):
                self.finish_reason = choice['finish_reason']
        return events

    def finish(self):
        if self.provider_models != {self.model}:
            raise ValueError('Chat provider served identifier differs from requested model')
        if self.finish_reason not in ('stop', 'tool_calls') or not self.saw_done:
            raise ValueError('Chat response lacks a complete provider finish boundary')
        events = []
        if self.reasoning:
            index, item = self.reasoning
            if self.encrypted:
                item['encrypted_content'] = self.encrypted
            events.append({'type': 'response.reasoning_summary_text.done', 'item_id': item['id'],
                           'output_index': index, 'summary_index': 0, 'text': item['summary'][0]['text']})
        for index, item in enumerate(self.items):
            item['status'] = 'completed'
            if item['type'] == 'function_call':
                # Parse before exposing completed arguments to the native CLI.
                json.loads(item['arguments'])
                if item['name'] in self.custom_tools:
                    item['type'] = 'custom_tool_call'
                    item['input'] = json.loads(item.pop('arguments'))['input']
                else:
                    events.append({'type': 'response.function_call_arguments.done',
                                   'item_id': item['id'], 'output_index': index,
                                   'arguments': item['arguments']})
            elif item['type'] == 'message':
                events.append({'type': 'response.output_text.done', 'item_id': item['id'],
                               'output_index': index, 'content_index': 0,
                               'text': item['content'][0]['text']})
            events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
        events.append({'type': 'response.completed', 'response': self.response('completed')})
        return events


def sse(event):
    return ('event: ' + event['type'] + '\ndata: ' + json.dumps(event, separators=(',', ':')) + '\n\n').encode()
