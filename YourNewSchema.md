# YNS - Your New Schema
> YNS is YNFO's schema validator.
> It utilizes the same exact syntax as YNFO, making writing schemas extreamily easy.
---

## How to start
To start using YNS you need to create a file with the extension `.yns`.
Once you created it you can start writing your schema.

---

This provides an extra layer of safety and predictibility to your data, a structure that will, and has to, be what you specified.
Like YNFO, YNS aims to be efficient and easy to read and write, you can look at a comparison between JSON Schema and YNS:

JSON Schema
```
{
  "type": "object",
  "properties": {
    "server": {
      "type": "object",
      "properties": {
        "ip": { "type": "string", "format": "ipv4" },
        "port": { "type": "integer", "default": 8080 },
        "timeout": { "type": "integer" }
      },
      "required": ["ip", "port"]
    },
    "features": {
      "type": "object",
      "properties": {
        "logging": { "type": "boolean" },
        "maxConnections": { "type": "integer", "minimum": 1, "maximum": 10000 }
      },
      "required": ["logging", "maxConnections"]
    }
  },
  "required": ["server", "features"]
}
```

YNS
```
.server :
    .ip [Ip]
    .port [Int] : 8080
    .timeout [Int]?

.features :
    .logging [Bool]
    .maxConnections [Int](1,10000)
```

---

## Optional Fields

A field can be set as optional by typing a question mark (`?`) right after the type of the field.
Like this:
`.optionalField [Int]?`

## Ranging values

You can specify a cerain range of values by using this syntax:
`.from_to [Int](0,100)`
In case of an optional field you can specify the range right after the question mark:
`.optionalFrom_to [Int]?(0,100)`

## Default values
You can set to a field a default value.
When you will make your YNFO, if a field with a default isn't present in your code then the parser will automatically add it to your output.
Using the same syntax of YNFO you can set a value right after a colon (`:`):
`.default [Int] : 123`

If an optional field has been given a default value, the parser will insert it in your output but it isn't required to be in it.
