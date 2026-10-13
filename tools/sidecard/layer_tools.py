# SPDX-License-Identifier: Apache-2.0
"""GLM checkpoint tool envelopes and OpenAI function-call interoperability."""

import json
import uuid

import regex as re

from .layer_reasoning import ReasoningStream

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
ARGUMENT = re.compile(r"\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
FUNCTION_NAME = re.compile(r"[A-Za-z0-9_.:-]+\Z")


def local_references(schema):
    """Schema validation must never fetch remote references from a client request."""
    pending = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for name in ("$ref", "$dynamicRef", "$recursiveRef"):
                if name in value and (not isinstance(value[name], str) or not value[name].startswith("#")):
                    raise ValueError("tool parameter references must be local to their schema")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def object_arguments(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as error:
            raise ValueError("function arguments must be a JSON object") from error
    if not isinstance(arguments, dict):
        raise ValueError("function arguments must be a JSON object")
    return arguments


def history_calls(calls):
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant tool_calls must be a nonempty list")
    result, identities = [], set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("only function tool calls are supported")
        function, identity = call.get("function"), call.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("tool call identifiers must be nonempty and unique")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("tool call function must have a name")
        if not FUNCTION_NAME.fullmatch(function["name"]):
            raise ValueError("invalid function name")
        identities.add(identity)
        result.append({**call, "function": {**function, "arguments": object_arguments(function.get("arguments"))}})
    return result


class Tools:
    def __init__(self, body):
        tools = body.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError("tools must be a list of function definitions")
        choice = body.get("tool_choice", "auto")
        if choice not in ("auto", "none"):
            raise ValueError("tool_choice currently supports auto or none")
        self.definitions, self.schemas = [], {}
        if tools:
            # The dependency is only needed for function-call requests.
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError
            from referencing.exceptions import Unresolvable

            self.resolution_error = Unresolvable

            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    raise ValueError("tools must contain only function definitions")
                function = tool.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError("tool function must have a name")
                name = function["name"]
                if not FUNCTION_NAME.fullmatch(name) or name in self.schemas:
                    raise ValueError("tool function names must be valid and unique")
                schema = function.get("parameters", {"type": "object", "properties": {}})
                if not isinstance(schema, dict):
                    raise ValueError("tool parameters must be a JSON schema object")
                local_references(schema)
                try:
                    Draft202012Validator.check_schema(schema)
                except SchemaError as error:
                    raise ValueError("invalid tool parameter schema") from error
                self.schemas[name] = (schema, Draft202012Validator(schema))
                self.definitions.append(tool)
        if choice == "none":
            self.definitions, self.schemas = [], {}

    def parse(self, block):
        name, separator, rest = block.partition("<arg_key>")
        name = name.strip()
        if name not in self.schemas:
            raise ValueError("generated tool call names an unregistered function")
        schema, validator = self.schemas[name]
        arguments = {}
        rest = "<arg_key>" + rest if separator else ""
        while rest.strip():
            match = ARGUMENT.match(rest)
            if match is None:
                raise ValueError("incomplete or malformed tool argument envelope")
            key, raw = match.groups()
            key = key.strip()
            if not key or key in arguments:
                raise ValueError("tool arguments must have unique names")
            if any(marker in raw for marker in ("<arg_key>", "</arg_key>", "<arg_value>")):
                raise ValueError("nested or unterminated tool argument envelope")
            parameter = schema.get("properties", {}).get(key, {})
            if parameter.get("type") == "string":
                value = raw
            else:
                try:
                    value = json.loads(raw)
                except ValueError:
                    value = raw
            arguments[key] = value
            rest = rest[match.end() :]
        try:
            valid = validator.is_valid(arguments)
        except self.resolution_error as error:
            raise ValueError("tool schema contains an unresolved reference") from error
        if not valid:
            raise ValueError("generated tool arguments do not match the registered schema")
        return {
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
            },
        }


class ToolStream:
    """Buffer only tool envelopes; release a call after complete schema validation."""

    def __init__(self, tools):
        self.tools, self.reasoning = tools, ReasoningStream()
        self.pending, self.in_tool, self.calls = "", False, []

    def feed(self, text, *, final=False):
        fields = self.reasoning.feed(text, final=final)
        content = fields.pop("content", "")
        if not self.tools.schemas:
            return {**fields, **({"content": content} if content else {})}
        self.pending += content
        emitted, calls = [], []
        while self.pending:
            marker = TOOL_CLOSE if self.in_tool else TOOL_OPEN
            before, found, after = self.pending.partition(marker)
            if not found:
                if final:
                    emitted.append((TOOL_OPEN if self.in_tool else "") + self.pending)
                    self.pending, self.in_tool = "", False
                elif not self.in_tool:
                    keep = max(
                        (size for size in range(1, len(marker)) if self.pending.endswith(marker[:size])), default=0
                    )
                    emitted.append(self.pending[:-keep] if keep else self.pending)
                    self.pending = self.pending[-keep:] if keep else ""
                break
            self.pending = after
            if self.in_tool:
                try:
                    call = self.tools.parse(before)
                except ValueError:
                    # Malformed output remains visible text; never dispatch it as a tool.
                    emitted.append(TOOL_OPEN + before + TOOL_CLOSE)
                else:
                    calls.append({"index": len(self.calls), **call})
                    self.calls.append(call)
            else:
                emitted.append(before)
            self.in_tool = not self.in_tool
        if emitted and "".join(emitted):
            fields["content"] = "".join(emitted)
        if calls:
            fields["tool_calls"] = calls
        return fields


def tool_message(text, tools):
    stream = ToolStream(tools)
    fields = stream.feed(text, final=True)
    fields.pop("tool_calls", None)
    if stream.calls:
        fields["tool_calls"] = stream.calls
    return {"role": "assistant", "content": fields.pop("content", ""), **fields}
