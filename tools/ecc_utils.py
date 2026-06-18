import torch


PRIMITIVE_POLYNOMIALS = {
    2: 0x7,
    3: 0xB,
    4: 0x13,
    5: 0x25,
    6: 0x43,
    7: 0x89,
    8: 0x11D,
    9: 0x211,
    10: 0x409,
}


def _trim(poly):
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _poly_mul_binary(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        if not av:
            continue
        for j, bv in enumerate(b):
            if bv:
                out[i + j] ^= 1
    return _trim(out)


class BinaryBCH:
    def __init__(self, m=8, t=26, primitive_polynomial=None):
        self.m = int(m)
        self.t = int(t)
        if self.m < 2:
            raise ValueError("BCH field degree m must be >= 2.")
        if self.t < 1:
            raise ValueError("BCH correction strength t must be >= 1.")
        self.n = (1 << self.m) - 1
        self.primitive_polynomial = int(
            primitive_polynomial or PRIMITIVE_POLYNOMIALS.get(self.m, 0)
        )
        if not self.primitive_polynomial:
            raise ValueError(f"No primitive polynomial configured for m={self.m}.")

        self.gf_exp, self.gf_log = self._build_field()
        self.generator = self._build_generator()
        self.generator_degree = len(self.generator) - 1
        self.k = self.n - self.generator_degree
        if self.k <= 0:
            raise ValueError(
                f"BCH(m={self.m}, t={self.t}) has non-positive k={self.k}; reduce t."
            )
        self.generator_desc = list(reversed(self.generator))

    def _build_field(self):
        n = self.n
        gf_exp = [0] * (2 * n + 1)
        gf_log = [0] * (n + 1)
        x = 1
        mask = 1 << self.m
        for i in range(n):
            gf_exp[i] = x
            gf_log[x] = i
            x <<= 1
            if x & mask:
                x ^= self.primitive_polynomial
            x &= n
        for i in range(n, len(gf_exp)):
            gf_exp[i] = gf_exp[i - n]
        return gf_exp, gf_log

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError("GF division by zero.")
        if a == 0:
            return 0
        return self.gf_exp[(self.gf_log[a] - self.gf_log[b]) % self.n]

    def _minimal_polynomial(self, exponent):
        coset = []
        seen = set()
        value = exponent % self.n
        while value not in seen:
            seen.add(value)
            coset.append(value)
            value = (value * 2) % self.n

        poly = [1]
        for item in coset:
            root = self.gf_exp[item]
            next_poly = [0] * (len(poly) + 1)
            for i, coeff in enumerate(poly):
                next_poly[i] ^= self.mul(coeff, root)
                next_poly[i + 1] ^= coeff
            poly = next_poly

        out = []
        for coeff in poly:
            if coeff not in (0, 1):
                raise ValueError(
                    f"Minimal polynomial coefficient is not binary: {coeff}"
                )
            out.append(int(coeff))
        return _trim(out), tuple(coset)

    def _build_generator(self):
        generator = [1]
        used_cosets = set()
        for exponent in range(1, 2 * self.t + 1):
            minimal, coset = self._minimal_polynomial(exponent)
            if coset[0] in used_cosets:
                continue
            used_cosets.update(coset)
            generator = _poly_mul_binary(generator, minimal)
        return generator

    def encode_block(self, message_bits):
        if len(message_bits) != self.k:
            raise ValueError(f"Expected {self.k} message bits, got {len(message_bits)}.")
        r = self.generator_degree
        work = [int(x) & 1 for x in message_bits] + [0] * r
        for i in range(self.k):
            if work[i]:
                for j, gv in enumerate(self.generator_desc):
                    work[i + j] ^= gv
        return [int(x) & 1 for x in message_bits] + work[self.k :]

    def encode(self, payload_bits):
        payload = [int(x) & 1 for x in payload_bits]
        if len(payload) % self.k != 0:
            raise ValueError("payload_bits length must be a multiple of BCH k.")
        out = []
        for start in range(0, len(payload), self.k):
            out.extend(self.encode_block(payload[start : start + self.k]))
        return out

    def _syndromes(self, received):
        syndromes = []
        for j in range(1, 2 * self.t + 1):
            value = 0
            for i, bit in enumerate(received):
                if int(bit) & 1:
                    exponent = (j * (self.n - 1 - i)) % self.n
                    value ^= self.gf_exp[exponent]
            syndromes.append(value)
        return syndromes

    def _berlekamp_massey(self, syndromes):
        c = [1]
        b = [1]
        l = 0
        shift = 1
        discrepancy_base = 1
        for n_idx in range(len(syndromes)):
            discrepancy = syndromes[n_idx]
            for i in range(1, l + 1):
                if i < len(c) and c[i] and syndromes[n_idx - i]:
                    discrepancy ^= self.mul(c[i], syndromes[n_idx - i])
            if discrepancy == 0:
                shift += 1
                continue

            old_c = c[:]
            coef = self.div(discrepancy, discrepancy_base)
            needed = len(b) + shift
            if len(c) < needed:
                c.extend([0] * (needed - len(c)))
            for i, bv in enumerate(b):
                if bv:
                    c[i + shift] ^= self.mul(coef, bv)

            if 2 * l <= n_idx:
                l = n_idx + 1 - l
                b = old_c
                discrepancy_base = discrepancy
                shift = 1
            else:
                shift += 1
        return _trim(c), l

    def _locator_value(self, locator, exponent):
        value = 0
        for i, coeff in enumerate(locator):
            if coeff:
                value ^= self.mul(coeff, self.gf_exp[(exponent * i) % self.n])
        return value

    def decode_block(self, received_bits):
        if len(received_bits) != self.n:
            raise ValueError(f"Expected {self.n} received bits, got {len(received_bits)}.")
        received = [int(x) & 1 for x in received_bits]
        syndromes = self._syndromes(received)
        if max(syndromes) == 0:
            return {
                "message_bits": received[: self.k],
                "corrected_codeword": received,
                "success": True,
                "corrected_errors": 0,
                "error_positions": [],
            }

        locator, degree = self._berlekamp_massey(syndromes)
        error_positions = []
        for i in range(self.n):
            symbol_exponent = self.n - 1 - i
            if self._locator_value(locator, (-symbol_exponent) % self.n) == 0:
                error_positions.append(i)

        corrected = received[:]
        for pos in error_positions:
            corrected[pos] ^= 1
        success = (
            len(error_positions) == degree
            and len(error_positions) <= self.t
            and max(self._syndromes(corrected)) == 0
        )
        return {
            "message_bits": corrected[: self.k],
            "corrected_codeword": corrected,
            "success": bool(success),
            "corrected_errors": int(len(error_positions)) if success else 0,
            "error_positions": error_positions,
        }

    def decode(self, coded_bits):
        coded = [int(x) & 1 for x in coded_bits]
        if len(coded) % self.n != 0:
            raise ValueError("coded_bits length must be a multiple of BCH n.")
        messages = []
        corrected = []
        failed = 0
        corrected_errors = 0
        block_results = []
        for start in range(0, len(coded), self.n):
            result = self.decode_block(coded[start : start + self.n])
            messages.extend(result["message_bits"])
            corrected.extend(result["corrected_codeword"])
            failed += 0 if result["success"] else 1
            corrected_errors += result["corrected_errors"]
            block_results.append(result)
        return {
            "message_bits": messages,
            "corrected_code_bits": corrected,
            "failed_codewords": failed,
            "corrected_bit_count": corrected_errors,
            "block_results": block_results,
        }


def make_bch_payload(num_available_bits, m, t, seed, device):
    codec = BinaryBCH(m=m, t=t)
    num_codewords = int(num_available_bits) // codec.n
    if num_codewords < 1:
        raise ValueError(
            f"Need at least {codec.n} selected positions for BCH, got {num_available_bits}."
        )
    payload_bits_count = num_codewords * codec.k
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    payload_bits = torch.randint(
        0,
        2,
        (payload_bits_count, 1),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    coded = codec.encode(payload_bits.reshape(-1).detach().cpu().tolist())
    coded_bits = torch.tensor(coded, device=device, dtype=torch.long).reshape(-1, 1)
    return codec, payload_bits, coded_bits


def bch_decode_metrics(decoded_coded_bits, payload_bits, m, t):
    codec = BinaryBCH(m=m, t=t)
    decoded_flat = decoded_coded_bits.reshape(-1).long().detach().cpu().tolist()
    payload_flat = payload_bits.reshape(-1).long().detach().cpu().tolist()
    result = codec.decode(decoded_flat)
    recovered = torch.tensor(result["message_bits"], dtype=torch.long).reshape(-1, 1)
    expected = torch.tensor(payload_flat, dtype=torch.long).reshape(-1, 1)
    matches = recovered == expected
    total_codewords = len(decoded_flat) // codec.n
    metrics = {
        "ecc": "bch",
        "bch_m": codec.m,
        "bch_t": codec.t,
        "bch_n": codec.n,
        "bch_k": codec.k,
        "bch_generator_degree": codec.generator_degree,
        "code_rate": float(codec.k / codec.n),
        "codewords": int(total_codewords),
        "effective_capacity_bits": int(expected.numel()),
        "effective_capacity_bytes": float(expected.numel() / 8.0),
        "payload_bit_accuracy": float(matches.float().mean().item()),
        "payload_bit_error_rate": float(1.0 - matches.float().mean().item()),
        "payload_bit_errors": int((~matches).sum().item()),
        "decode_success": bool(result["failed_codewords"] == 0 and bool(matches.all().item())),
        "failed_codewords": int(result["failed_codewords"]),
        "uncorrectable_codeword_count": int(result["failed_codewords"]),
        "corrected_bit_count": int(result["corrected_bit_count"]),
    }
    return recovered, torch.tensor(result["corrected_code_bits"], dtype=torch.long).reshape(-1, 1), metrics


def bch_settings(codec, num_codewords, used_bits, ecc_seed):
    return {
        "ecc": "bch",
        "bch_m": int(codec.m),
        "bch_t": int(codec.t),
        "bch_n": int(codec.n),
        "bch_k": int(codec.k),
        "bch_generator_degree": int(codec.generator_degree),
        "code_rate": float(codec.k / codec.n),
        "codewords": int(num_codewords),
        "used_coded_bits": int(used_bits),
        "effective_capacity_bits": int(num_codewords * codec.k),
        "effective_capacity_bytes": float((num_codewords * codec.k) / 8.0),
        "ecc_seed": int(ecc_seed),
    }
