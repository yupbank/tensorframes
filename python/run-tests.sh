#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# assumes run from python/ directory
if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

if [ -z "$SCALA_BINARY_VERSION" ]; then
    echo 'You need to set $SCALA_BINARY_VERSION (2.10, 2.11, ...) to run these tests.' >&2
    exit 1
fi


LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PROJECT_HOME="$DIR/../"


# Tests run with scala-2.10 because the binaries are distributed in scala 2.10 for now.
JAR_PATH="$PROJECT_HOME/target/scala-$SCALA_BINARY_VERSION/tensorframes-assembly-0.2.2.jar"

export PYSPARK_SUBMIT_ARGS="--jars $JAR_PATH pyspark-shell"

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.

export PYTHONPATH=$PYTHONPATH:"$PROJECT_HOME/src/main/python/"

# Run test suites

nosetests -v --all-modules -w "$PROJECT_HOME/src/main/python"


# Run doc tests
# No run of doc tests for now
