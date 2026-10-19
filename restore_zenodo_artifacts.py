#!/usr/bin/env python3
"""Restore the byte-identical ProMeta artifact ZIP from Zenodo volumes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MANIFEST = json.loads(r'''{
  "format": "byte-concatenation",
  "original": {
    "name": "prometa_artifacts_20261019_cdk6.zip",
    "bytes": 987338969,
    "sha256": "a42dbf02943b96394d9da8fff4b790f9995f9793f04b40d696757e0cf9d3f4fd",
    "md5": "db0c73ba8503385af897d011874328bb"
  },
  "zenodo_record_id": 23474832,
  "parts": [
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part001",
      "bytes": 33554432,
      "sha256": "f11e6b6d878954a61572c3d59eda3deec856f53b92e2cef799eecb6d9cb7dc65",
      "md5": "2072c9a25ad406ba2bcab27e9ec4605c"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part002",
      "bytes": 33554432,
      "sha256": "62f331aae7eb995370916b3212719880fa54eab94c09b2fd20b8fd8398367271",
      "md5": "b20d8cd42294f24a5cc590624d92c4d1"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part003",
      "bytes": 33554432,
      "sha256": "54f19f765e70663cd62dd16e36324ad252a93d929969ee5579524a031ba45ac5",
      "md5": "5e6e52d2297d5c71b638c569b4227441"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part004",
      "bytes": 33554432,
      "sha256": "a9a42d0f05c474653208c0c3db16330ff49b0bef9a5f04ba1031120e6b664a2a",
      "md5": "52fcfca5589c584c434f86e5bb3036ad"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part005",
      "bytes": 33554432,
      "sha256": "6be7a64450f077ff6976abe8fa2a09e4c52389d4d51d1f8f7c8c3f51187e1c4c",
      "md5": "a77a9595f7cbe694c142da3d608406f1"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part006",
      "bytes": 33554432,
      "sha256": "23dfcf668d76356d2d66383cf27c89a2dd75b443ca4f928ab900997df00cf08c",
      "md5": "4a2e7f110a9897d3dfdc7ed8dc200fab"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part007",
      "bytes": 33554432,
      "sha256": "c45ef472fab59045879dc0b33e8cf1cf8be7badbc7d39da2b083cabd9cfa8ffe",
      "md5": "94e16fd45f8527495a560c95151fc3e1"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part008",
      "bytes": 33554432,
      "sha256": "a023d4955d731f22af3e3a49851f96af7664765307e98cec8e11773955d49450",
      "md5": "b066a25198b4fbf414e94a175add9050"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part009",
      "bytes": 33554432,
      "sha256": "fb0abdabb9a0c7d2b6d5a197e689edac23182e9b6be0d25a7b836b789d13941d",
      "md5": "cd6cf89004a518097e1c55ea6da6fef3"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part010",
      "bytes": 33554432,
      "sha256": "0dbe25291f7e9137996f34f5bafd8d23b19886d29db5aeeee5ee9d4829951a78",
      "md5": "707702ce2572ccbd04a30fe7b607721b"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part011",
      "bytes": 33554432,
      "sha256": "e52f0c0bc1421bbef078057160d902454706a954a3225f4405a871542a38dd19",
      "md5": "d1bb19314e9982afa04941e7ef5ea8c5"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part012",
      "bytes": 33554432,
      "sha256": "d2e7292ca932dc663dcd6359a126c07838a461b2f6bc003f8799f91e1164f961",
      "md5": "43694412c28302a65db7a90b17941f33"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part013",
      "bytes": 33554432,
      "sha256": "3a37d3b969b79598667c03e19a81f85d6ca76eb4669a861d379a178937bac0c3",
      "md5": "d54e647b1f40a993da54a70ea8304d6b"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part014",
      "bytes": 33554432,
      "sha256": "9cdaff876322c9d9d3dcebee3522d07f8ccc6d74fa9caeacd62086dce2344ead",
      "md5": "64351f43f6f84ee0c8728b988e85f0ab"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part015",
      "bytes": 33554432,
      "sha256": "9484fbb67fd6388d5cd616b0dd06b390712cfb6f794b7f80f728c15e99df4d96",
      "md5": "deb395042f06e19bef709381506080b9"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part016",
      "bytes": 33554432,
      "sha256": "b90ef8dc25a8a9901c92756278a9052d43eb546bc6af12661af0323903e94409",
      "md5": "093d1a99c0f7d9af19c84e831574be7e"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part017",
      "bytes": 33554432,
      "sha256": "7a6c4cae7a8e38b50ca988d65fad9840f93115596429242361c7e923fc9f6cdc",
      "md5": "148f09cddda4397bf30aa74a62108f51"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part018",
      "bytes": 33554432,
      "sha256": "206585048be9c53dba69d37c7a72d060728c5a36519493d12197923e1ed517d4",
      "md5": "3fded5a3b70b5544f5360512b553df11"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part019",
      "bytes": 33554432,
      "sha256": "a9c0940f635ae64965b285c27b8914138eab4927c286f63296d250ba46b21745",
      "md5": "cec9ece38925e80375e17d66cd284d45"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part020",
      "bytes": 33554432,
      "sha256": "54fd7269db09e37dbf9a49acd0ec9947ae103dd7f90123762bfe677c40cfaae8",
      "md5": "298939ce31f312246cb8f16b585581d7"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part021",
      "bytes": 33554432,
      "sha256": "f73696d7d76aa8c6dca97e33cb4b85d6e9d7a79d0733d94ed838e8dd5fe8e2f6",
      "md5": "08e0470e84bf2591ecf3c35cb98c9e51"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part022",
      "bytes": 33554432,
      "sha256": "a7037fc785fd2a12013a3983171d8c797b3523db83971403634ee466c4149d88",
      "md5": "086acb74f337f9d24e1bdd138a8ab0f3"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part023",
      "bytes": 33554432,
      "sha256": "4270f9eba5a4ba4900298fea384dc91a88f384c57214883ef1afb232230ba568",
      "md5": "c9089f0799c59ccec786f006604b4897"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part024",
      "bytes": 33554432,
      "sha256": "e84fd1ea04bab646ca618fc0905b3d9abb3f17fb58097581c72e4e7ad1397e7e",
      "md5": "24df4d0f41521e89be4a57e5848f86e2"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part025",
      "bytes": 33554432,
      "sha256": "51dfb62dbf80246ee823ef5dda9ef50d07418c8438ff008a01a1b78dba35258e",
      "md5": "80f6ffe1037adede6ee376cda44d8edf"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part026",
      "bytes": 33554432,
      "sha256": "af8763fe8f58ed0a92dd4f92dbc58fb637d75ad148539a754695f350c221d598",
      "md5": "d8ed8a50a1a2d2ec17972712a9d46e46"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part027",
      "bytes": 33554432,
      "sha256": "086ba5699d6715399a05c7961a2cb8bf4282fcb1947b94d0bdfc81654a85430b",
      "md5": "190cb8893425f9aa539d52ce118ba53c"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part028",
      "bytes": 33554432,
      "sha256": "9a8b768307bca791d90d0783b77bba3cdd5e9c974fb13075b2961f6437fc7540",
      "md5": "b706d899d60d08165222c0fb7b35ceaa"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part029",
      "bytes": 33554432,
      "sha256": "e0c3f18637a4c3023ccdd0b83c9f186355d3f80b4a6d284c09dd047d97fd0d99",
      "md5": "9157256620a7f91458673e2ed39654c9"
    },
    {
      "name": "prometa_artifacts_20261019_cdk6.zip.part030",
      "bytes": 14260441,
      "sha256": "984daeb12c769fc5b41e3a957916a7743444b98d6fb7e7cbdc0a8a395c6dc02b",
      "md5": "980706dfd247b6148c242ff4c43d9bdf"
    }
  ]
}''')


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("."), help="Directory containing the 30 volumes")
    parser.add_argument("--output", type=Path, help="Restored ZIP path (default: original filename in current directory)")
    parser.add_argument("--download", action="store_true", help="Download any missing volumes from the public Zenodo archive")
    args = parser.parse_args()
    output = args.output or Path(MANIFEST["original"]["name"])
    original = MANIFEST["original"]
    if output.exists():
        if output.stat().st_size == original["bytes"] and digest(output) == original["sha256"]:
            print("Already verified:", output)
            return
        raise SystemExit("Output exists with different content; choose another --output path.")
    args.directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in MANIFEST["parts"]:
        path = args.directory / part["name"]
        if not path.exists():
            if not args.download:
                raise SystemExit("Missing " + str(path) + "; download all volumes or use --download.")
            url = "https://zenodo.org/api/records/" + str(MANIFEST["zenodo_record_id"]) + "/files/" + part["name"] + "/content"
            temporary = path.with_suffix(path.suffix + ".downloading")
            print("Downloading:", part["name"], flush=True)
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size != part["bytes"] or digest(temporary) != part["sha256"]:
                raise SystemExit("Downloaded volume checksum mismatch: " + part["name"])
            temporary.replace(path)
        if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
            raise SystemExit("Volume checksum mismatch: " + str(path))
        paths.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    h = hashlib.sha256()
    with temporary.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    h.update(block)
    if temporary.stat().st_size != original["bytes"] or h.hexdigest() != original["sha256"]:
        raise SystemExit("Restored ZIP checksum mismatch.")
    temporary.replace(output)
    print("PASS: restored ZIP matches the GitHub original byte for byte:", output)
    print("SHA-256:", h.hexdigest())
    print("Extract this ZIP inside the code repository, then run: python verify_release.py")


if __name__ == "__main__":
    main()
