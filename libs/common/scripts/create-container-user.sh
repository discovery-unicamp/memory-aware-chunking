#!/bin/sh

set -e

user_name=experiment
group_name=experiment

echo "Creating user with UID=${HOST_UID}, GID=${HOST_GID}"

existing_group=$(getent group "${HOST_GID}" | cut -d: -f1)
if [ -z "$existing_group" ]; then
  echo "Creating group $group_name with GID ${HOST_GID}"
  addgroup --gid "${HOST_GID}" "$group_name"
else
  echo "Group with GID ${HOST_GID} already exists: $existing_group"
  group_name=$existing_group
fi

existing_user=$(getent passwd "${HOST_UID}" | cut -d: -f1)
if [ -z "$existing_user" ]; then
  echo "Creating user $user_name with UID ${HOST_UID} and GID ${HOST_GID}"
  adduser --uid "${HOST_UID}" --gid "${HOST_GID}" --disabled-password --gecos "" "$user_name"
else
  echo "User with UID ${HOST_UID} already exists: $existing_user"
  usermod -g "${HOST_GID}" "$existing_user"
  user_name=$existing_user
fi

mkdir -p /experiment
chown "${HOST_UID}:${HOST_GID}" /experiment