# YNFO - Your New Format

> YNFO (pronounced just like "info") is a lightweight and versitile data format.
> It was made as lighter alternative to more popular formats such as: JSON, YAML and TOML.

This format allows for:
- Comments
- Referencing fields from other files
- Flexible syntax

## Fields
A field, in terms of JavaScript, can be compared to an object, where it can contain:
- Other fields
- Single elements
- Or alternattively both

A field is defined just like below:
```
.myField : "myValue"
```
Fields are dynamically typed and support integers, floats, strings and booleans.
Everything that isn't recognized as any of those is automatically set as a string.

## Lists
A list is a field with multiple elements separated by any whitespace.
An element is a single number (integer or float), a boolean value (true or false) or any character that is enclosed in double quotes (a string).
In this sense a list can be written like this:
```
.list :
  1
  2
  3
```
or like this:
```
.list : 1 2 3
```

Both give the same result.

## Fields within fields
As we said above, a field can also contain an other or more fields.
The syntax is similar to lists:
```
.myField :
  .mySecondField : 3.14
  .myThirdField : true
```

## Referencing
It's also possible to reference fields from other files.

Let's say that we have a file called `server_info.ynfo`:
```
.ip : 192.335.9.60
.port : 8080
.users:
  "user1"
  "user2"
  "user3"
.admin :
  .password : "admin_password"
```
Inside of a second file we can reference these fields as values:
```
.ip : server_info.ip
.port : server_info.port
.favouriteUser : server_info.users[1]
.adminPwd : server_info.admin.password
```

You can also reference fields within the same file by either writing the name of the file or using `self`, for example:
```
self.admin.password
```

## Comments
YNFO has support for inline comments.
```
<this is a comment>
```
